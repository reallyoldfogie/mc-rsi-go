package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// runConfigFlagName is -run-config's own name, factored out once since
// scanRunConfigFlag and applyRunConfigFile (and the registration in
// parseFlags) all need to agree on it, and applyRunConfigFile also needs
// it to reject recursive use from inside a file - see that function's
// own doc comment.
const runConfigFlagName = "run-config"

// scanRunConfigFlag finds -run-config's value in args, if present,
// without requiring every other flag in args to already be registered
// anywhere - unlike running a throwaway single-flag FlagSet over args
// (which would stop at the first *other* flag it doesn't recognize,
// missing -run-config entirely whenever something else precedes it),
// this walks every token looking specifically for -run-config/
// --run-config and ignores everything else, so it works regardless of
// where in args it appears. See docs/plans/22-run-config-file.md's
// "Loading mechanism" for why this needs to run before the real
// fs.Parse(args) in parseFlags: that's what lets a run-config file
// supply *defaults* that an explicit command-line flag still overrides.
func scanRunConfigFlag(args []string) (string, error) {
	for i := 0; i < len(args); i++ {
		name, hasValue, value := splitFlagArg(args[i])
		if name != runConfigFlagName {
			continue
		}
		if hasValue {
			return value, nil
		}
		if i+1 >= len(args) {
			return "", fmt.Errorf("rsi-train: -%s needs a value", runConfigFlagName)
		}
		return args[i+1], nil
	}
	return "", nil
}

// splitFlagArg recognizes just enough of flag.Parse's own syntax (one or
// two leading dashes, an optional "=value" suffix) to find a specific
// flag by name without parsing anything else in arg - a plain positional
// token (most commonly some other flag's own value, e.g. the "10" in
// "-eval-episodes 10") never starts with '-' and is reported as not a
// flag at all (name == "").
func splitFlagArg(arg string) (name string, hasValue bool, value string) {
	if len(arg) < 2 || arg[0] != '-' {
		return "", false, ""
	}
	trimmed := strings.TrimPrefix(strings.TrimPrefix(arg, "-"), "-")
	if eq := strings.IndexByte(trimmed, '='); eq >= 0 {
		return trimmed[:eq], true, trimmed[eq+1:]
	}
	return trimmed, false, ""
}

// applyRunConfigFile reads path as a flat JSON object and applies each
// key as that same-named flag's new default, via fs.Set - the same
// mechanism fs.Parse itself uses to assign a flag's value, so calling
// this BEFORE fs.Parse(args) in parseFlags makes "the file supplies
// defaults, the command line overrides them" true for free: whichever
// assignment happens second (fs.Parse, for anything actually given on
// the command line) simply wins, since both just write into the same
// already-registered flag.Value. See runConfigValueToString below for
// how a decoded JSON value becomes the string fs.Set itself expects.
//
// Every key must name an already-registered flag (fs.Lookup) - an
// unknown key fails fast rather than silently doing nothing, since
// catching a typo is this feature's entire point. Two exceptions:
//   - a key starting with "_" is skipped entirely, treated as a
//     documentation-only comment - JSON has no native comment syntax,
//     and this project's own curriculum.json/trainer-config.json files
//     already use a leading-underscore "_comment" key for the same
//     purpose, so run-config files follow that existing convention
//     rather than inventing a different one.
//   - the key "run-config" itself is rejected outright (not silently
//     ignored) - recursion was considered and deliberately not
//     supported (docs/plans/22-run-config-file.md's own Open
//     questions), so a file setting this key is almost certainly a
//     mistake worth surfacing, not something to honor or quietly drop.
func applyRunConfigFile(fs *flag.FlagSet, path string) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("reading %s: %w", path, err)
	}
	var values map[string]any
	if err := json.Unmarshal(raw, &values); err != nil {
		return fmt.Errorf("parsing %s as JSON: %w", path, err)
	}
	for key, value := range values {
		if strings.HasPrefix(key, "_") {
			continue
		}
		if key == runConfigFlagName {
			return fmt.Errorf("%s: %q is not allowed inside a run-config file (no recursion)", path, key)
		}
		target := fs.Lookup(key)
		if target == nil {
			return fmt.Errorf("%s: unknown flag name %q", path, key)
		}
		str, err := runConfigValueToString(value)
		if err != nil {
			return fmt.Errorf("%s: %q: %w", path, key, err)
		}
		if err := target.Value.Set(str); err != nil {
			return fmt.Errorf("%s: %q: %w", path, key, err)
		}
	}
	return nil
}

// runConfigValueToString converts one decoded JSON value to the string
// form flag.Value.Set expects. encoding/json always decodes a JSON
// number into float64 (there is no separate JSON int type), so an
// IntVar flag's value (e.g. 8) arrives here as the float64 8.0 - naively
// formatting that with fmt.Sprintf("%v", ...) gives "8", which happens
// to work, but relying on %v's own default float formatting is fragile
// (it switches to %g / exponent notation for sufficiently large values).
// strconv.FormatFloat with prec=-1 instead gives the shortest decimal
// string that round-trips to the exact same float64, which for every
// whole number this project's flags actually use (8, 300, 15, ...) is
// just that integer with no trailing ".0" - verified directly, see this
// function's own tests - and for a genuine fraction (0.2, 0.19) gives
// the expected decimal form, so one call correctly handles both an
// IntVar and a Float64Var target without needing to know which.
func runConfigValueToString(value any) (string, error) {
	switch v := value.(type) {
	case bool:
		return strconv.FormatBool(v), nil
	case string:
		return v, nil
	case float64:
		return strconv.FormatFloat(v, 'f', -1, 64), nil
	default:
		return "", fmt.Errorf("unsupported value type %T (expected a bool, number, or string)", value)
	}
}
