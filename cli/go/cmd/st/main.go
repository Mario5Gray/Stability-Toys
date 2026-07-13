// Command st is the Stability-Toys operations CLI. It is the user-facing surface
// over pkg/stclient: generation (WS), reads (HTTP), uploads, super-resolution,
// and job control. The backend is remote; point it at one with --server or
// $ST_SERVER.
package main

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"time"

	"github.com/spf13/cobra"

	"github.com/darkbit/stability-toys/cli/st/internal/config"
	"github.com/darkbit/stability-toys/cli/st/internal/history"
	"github.com/darkbit/stability-toys/cli/st/pkg/stclient"
)

// Global flags, bound to the root command's persistent flag set.
var (
	flagServer    string
	flagConfig    string
	flagOutputDir string
	flagJSON      bool
	flagTimeout   time.Duration
)

var rootCmd = &cobra.Command{
	Use:           "st",
	Short:         "Stability-Toys operations CLI",
	SilenceUsage:  true,
	SilenceErrors: true,
}

var (
	resolveStateRoot = history.ResolveStateRoot
	newHistoryStore  = func(root string) history.Store { return history.NewFSStore(root) }
)

func loadStateStore() (history.Store, error) {
	root, err := resolveStateRoot()
	if err != nil {
		return nil, err
	}
	return newHistoryStore(root), nil
}

func init() {
	pf := rootCmd.PersistentFlags()
	pf.StringVar(&flagServer, "server", os.Getenv("ST_SERVER"), "backend base URL (or $ST_SERVER)")
	pf.StringVar(&flagConfig, "config", "", "config file path (or $ST_CONFIG, then XDG default)")
	pf.StringVarP(&flagOutputDir, "output-dir", "o", "", "directory for generated images (overrides config)")
	pf.BoolVar(&flagJSON, "json", false, "emit machine-readable JSON")
	pf.DurationVar(&flagTimeout, "timeout", 0, "per-request timeout (0 = client default)")
}

// resolveConfig loads the config at path, or writes a template and reports the
// path when none exists. A nil cfg with bootstrapped=true means "tell the user
// to edit the template and re-run".
func resolveConfig(path string) (cfg *config.Config, message string, bootstrapped bool) {
	if _, err := os.Stat(path); os.IsNotExist(err) {
		_ = config.BootstrapTemplate(path)
		return nil, fmt.Sprintf("No config found. Wrote a template to %s — edit output_directory/meta and re-run.", path), true
	}
	c, err := config.Load(path)
	if err != nil {
		return nil, fmt.Sprintf("config %s: %v", path, err), false
	}
	return c, "", false
}

// requireConfig resolves the config path (--config > $ST_CONFIG > XDG) and gates
// on it. On first run it bootstraps a template and exits non-zero so the user can
// edit it. Config-dependent commands (e.g. gen) call this in their RunE; commands
// that need no config (modes, upload, …) skip it.
func requireConfig() (*config.Config, error) {
	path, err := config.Resolve(flagConfig)
	if err != nil {
		return nil, err
	}
	cfg, message, bootstrapped := resolveConfig(path)
	if bootstrapped {
		return nil, exitError{code: 2, err: fmt.Errorf("%s", message)}
	}
	if cfg == nil {
		return nil, fmt.Errorf("%s", message)
	}
	return cfg, nil
}

// resolveServerURL returns the server base URL using precedence:
// explicit serverFlag > config file server_url. Returns empty string
// when neither is set; callers (stclient.New) handle the empty case.
func resolveServerURL(serverFlag, configFlag string) string {
	if serverFlag != "" {
		return serverFlag
	}
	path, err := config.Resolve(configFlag)
	if err != nil {
		return ""
	}
	cfg, err := config.Load(path)
	if err != nil {
		return ""
	}
	return cfg.ServerURL
}

// newClient builds an stclient pointed at the resolved server URL, honoring
// --timeout when set. URL precedence: --server/$ST_SERVER > config server_url.
func newClient() *stclient.Client {
	var opts []stclient.Option
	if flagTimeout > 0 {
		opts = append(opts, stclient.WithHTTPClient(&http.Client{Timeout: flagTimeout}))
	}
	return stclient.New(resolveServerURL(flagServer, flagConfig), opts...)
}

func main() {
	if err := executeCLI(context.Background(), os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(exitCodeOf(err))
	}
}
