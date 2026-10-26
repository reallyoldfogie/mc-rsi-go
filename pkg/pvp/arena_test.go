package pvp

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/reallyoldfogie/mc-agent/models"
	"github.com/reallyoldfogie/mc-client-test-go/testenv"
)

// fakeRCON is a minimal testenv.RCON backing a real testenv.RCONHelper via
// testenv.NewRCONHelper - the same approach mc-agent's own test fakes use
// for the lower-level RCON surface, letting Teleport/GetEntityPos/etc.
// run their real parsing logic against scripted command/response pairs.
type fakeRCON struct {
	mu        sync.Mutex
	responses map[string]string
	cmds      []string
}

func (f *fakeRCON) Exec(ctx context.Context, cmd string) (string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.cmds = append(f.cmds, cmd)
	if resp, ok := f.responses[cmd]; ok {
		return resp, nil
	}
	return "", nil
}

func (f *fakeRCON) Close() error                        { return nil }
func (f *fakeRCON) Reconnect(ctx context.Context) error { return nil }

func (f *fakeRCON) commands() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.cmds...)
}

func newFakeRCONHelper(responses map[string]string) (testenv.RCONHelper, *fakeRCON) {
	f := &fakeRCON{responses: responses}
	return testenv.NewRCONHelper(f), f
}

// fakeController is a scripted RCONMatchController recording every call.
type fakeController struct {
	mu            sync.Mutex
	resetCalls    []string
	teamCalls     map[string][]string
	verifyResults map[string]bool
	resetErr      error
}

func (c *fakeController) VerifyPlayerNearPosition(ctx context.Context, playerName string, center models.V3, radius float64) (bool, error) {
	if c.verifyResults == nil {
		return true, nil
	}
	return c.verifyResults[playerName], nil
}

func (c *fakeController) SeedMatchReset(ctx context.Context, playerName, weapon string, center models.V3, radius float64) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.resetCalls = append(c.resetCalls, playerName)
	return c.resetErr
}

func (c *fakeController) SeedMatchTeam(ctx context.Context, teamName string, memberNames []string, center models.V3, radius float64) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.teamCalls == nil {
		c.teamCalls = make(map[string][]string)
	}
	c.teamCalls[teamName] = append([]string(nil), memberNames...)
	return nil
}

func TestMatchArena_ResetMatchHealsAndTeleportsEveryParticipant(t *testing.T) {
	controller := &fakeController{}
	rcon, fake := newFakeRCONHelper(nil)
	arena := MatchArena{
		Controller:       controller,
		RCON:             rcon,
		Weapon:           "minecraft:iron_sword",
		Center:           models.V3{X: 10, Y: 64, Z: 10},
		Radius:           20,
		ParticipantNames: []string{"Alice", "Bob"},
	}

	if err := arena.ResetMatch(context.Background()); err != nil {
		t.Fatalf("ResetMatch: %v", err)
	}

	if len(controller.resetCalls) != 2 || controller.resetCalls[0] != "Alice" || controller.resetCalls[1] != "Bob" {
		t.Fatalf("resetCalls = %v, want [Alice Bob]", controller.resetCalls)
	}

	wantTeleports := []string{"tp Alice 10.00 64.00 10.00", "tp Bob 10.00 64.00 10.00"}
	for _, want := range wantTeleports {
		found := false
		for _, cmd := range fake.commands() {
			if cmd == want {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("commands = %v, want to include %q", fake.commands(), want)
		}
	}
}

func TestMatchArena_ResetMatchPropagatesControllerError(t *testing.T) {
	wantErr := fmt.Errorf("boom")
	controller := &fakeController{resetErr: wantErr}
	rcon, _ := newFakeRCONHelper(nil)
	arena := MatchArena{Controller: controller, RCON: rcon, ParticipantNames: []string{"Alice"}}
	err := arena.ResetMatch(context.Background())
	if err == nil || !strings.Contains(err.Error(), "boom") {
		t.Fatalf("ResetMatch error = %v, want wrapping %v", err, wantErr)
	}
}

func TestFormTeams_GroupsByTeamAndSkipsFreeForAll(t *testing.T) {
	controller := &fakeController{}
	participants := []Participant{
		{ID: "red1", Team: "Red"},
		{ID: "red2", Team: "Red"},
		{ID: "blue1", Team: "Blue"},
		{ID: "ffa", Team: ""},
	}
	names := map[string]string{"red1": "Alice", "red2": "Bob", "blue1": "Carol", "ffa": "Dave"}

	if err := FormTeams(context.Background(), controller, participants, names, models.V3{}, 10); err != nil {
		t.Fatalf("FormTeams: %v", err)
	}

	if len(controller.teamCalls) != 2 {
		t.Fatalf("teamCalls = %v, want exactly 2 teams", controller.teamCalls)
	}
	if got := controller.teamCalls["Red"]; len(got) != 2 {
		t.Errorf("Red team members = %v, want 2", got)
	}
	if got := controller.teamCalls["Blue"]; len(got) != 1 || got[0] != "Carol" {
		t.Errorf("Blue team members = %v, want [Carol]", got)
	}
	if _, ok := controller.teamCalls[""]; ok {
		t.Error("free-for-all participants must not form a team")
	}
}

func TestMatchOrigin_ZeroForFirstMatchNonZeroForOthers(t *testing.T) {
	base := models.V3{X: 100, Y: 64, Z: 100}
	first := MatchOrigin(base, 0, 16)
	if first != base {
		t.Errorf("MatchOrigin(base, 0, ...) = %+v, want base unchanged %+v", first, base)
	}
	second := MatchOrigin(base, 1, 16)
	if second == base {
		t.Error("MatchOrigin(base, 1, ...) must differ from match 0's origin")
	}
	if second.Y != base.Y || second.Z != base.Z {
		t.Errorf("MatchOrigin should only offset X, got %+v", second)
	}
}

func TestEnforceContainment_TeleportsBackWhenOutsideRadius(t *testing.T) {
	responses := map[string]string{
		"data get entity Wanderer Pos": "Wanderer has the following entity data: [500.0d, 64.0d, 500.0d]",
	}
	rcon, fake := newFakeRCONHelper(responses)
	ctx, cancel := context.WithTimeout(context.Background(), 150*time.Millisecond)
	defer cancel()

	EnforceContainment(ctx, rcon, []string{"Wanderer"}, models.V3{X: 0, Y: 64, Z: 0}, 10, 20*time.Millisecond)

	found := false
	for _, cmd := range fake.commands() {
		if cmd == "tp Wanderer 0.00 64.00 0.00" {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("commands = %v, want a teleport back to center", fake.commands())
	}
}

func TestEnforceContainment_LeavesParticipantInsideRadiusAlone(t *testing.T) {
	responses := map[string]string{
		"data get entity Homebody Pos": "Homebody has the following entity data: [1.0d, 64.0d, 1.0d]",
	}
	rcon, fake := newFakeRCONHelper(responses)
	ctx, cancel := context.WithTimeout(context.Background(), 150*time.Millisecond)
	defer cancel()

	EnforceContainment(ctx, rcon, []string{"Homebody"}, models.V3{X: 0, Y: 64, Z: 0}, 10, 20*time.Millisecond)

	for _, cmd := range fake.commands() {
		if strings.HasPrefix(cmd, "tp Homebody") {
			t.Fatalf("Homebody is within radius, should not have been teleported; commands = %v", fake.commands())
		}
	}
}

// blockingFirstCallRCON is a testenv.RCON whose Exec call for
// blockOnSubstring blocks until released, so a test can cancel ctx while
// that call is still "in flight" and confirm a caller checks ctx before
// issuing its next call, rather than only between ticks. Every other
// call (including testenv.NewRCONHelper's own hidden version-detection
// probe on construction) returns immediately.
type blockingFirstCallRCON struct {
	mu               sync.Mutex
	cmds             []string
	released         chan struct{}
	blockOnSubstring string
}

func (f *blockingFirstCallRCON) Exec(ctx context.Context, cmd string) (string, error) {
	f.mu.Lock()
	f.cmds = append(f.cmds, cmd)
	shouldBlock := strings.Contains(cmd, f.blockOnSubstring)
	f.mu.Unlock()
	if shouldBlock {
		<-f.released
	}
	return "", nil
}

func (f *blockingFirstCallRCON) Close() error                        { return nil }
func (f *blockingFirstCallRCON) Reconnect(ctx context.Context) error { return nil }

func (f *blockingFirstCallRCON) commands() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.cmds...)
}

func TestEnforceContainment_StopsBetweenNamesOnceContextCanceled(t *testing.T) {
	fake := &blockingFirstCallRCON{released: make(chan struct{}), blockOnSubstring: "Alice"}
	rcon := testenv.NewRCONHelper(fake)
	ctx, cancel := context.WithCancel(context.Background())

	done := make(chan struct{})
	go func() {
		EnforceContainment(ctx, rcon, []string{"Alice", "Bob"}, models.V3{}, 10, time.Hour)
		close(done)
	}()

	// Give EnforceContainment's first tick time to start its first
	// GetEntityPos call (which blocks in blockingFirstCallRCON until
	// released), then cancel ctx while that call is still outstanding.
	time.Sleep(50 * time.Millisecond)
	cancel()
	time.Sleep(50 * time.Millisecond)
	close(fake.released)

	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("EnforceContainment did not return after ctx was canceled")
	}

	for _, cmd := range fake.commands() {
		if strings.Contains(cmd, "Bob") {
			t.Fatalf("Bob should never have been queried once ctx was canceled mid-tick; commands = %v", fake.commands())
		}
	}
}

func TestSetWorldBorder_IssuesCenterAndSizeCommands(t *testing.T) {
	rcon, fake := newFakeRCONHelper(nil)
	if err := SetWorldBorder(context.Background(), rcon, models.V3{X: 5, Y: 64, Z: -5}, 50); err != nil {
		t.Fatalf("SetWorldBorder: %v", err)
	}
	// newFakeRCONHelper's underlying testenv.NewRCONHelper issues its own
	// "version" auto-detect probe on construction, so don't assert an
	// exact command count/order - just that both expected commands appear.
	cmds := fake.commands()
	want := []string{"worldborder center 5.000000 -5.000000", "worldborder set 100.000000"}
	for _, wantCmd := range want {
		found := false
		for _, cmd := range cmds {
			if cmd == wantCmd {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("commands = %v, want to include %q", cmds, wantCmd)
		}
	}
}
