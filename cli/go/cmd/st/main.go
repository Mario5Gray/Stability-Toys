// Command st is the Stability-Toys operations CLI. It is the user-facing surface
// over pkg/stclient: generation (WS), reads (HTTP), uploads, super-resolution,
// and job control. The backend is remote; point it at one with --server or
// $ST_SERVER.
package main

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"strings"
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

// configFlagHelp names the resolved default path. XDG_CONFIG_HOME is usually
// unset, so "the XDG default" alone tells a new user nothing they can act on.
func configFlagHelp() string {
	def, err := config.Resolve("")
	if err != nil || def == "" {
		return "config file path (or $ST_CONFIG)"
	}
	return fmt.Sprintf("config file path (or $ST_CONFIG; default %s -- run 'st config' to see all paths)", def)
}

func init() {
	pf := rootCmd.PersistentFlags()
	pf.StringVar(&flagServer, "server", os.Getenv("ST_SERVER"), "backend base URL (or $ST_SERVER)")
	pf.StringVar(&flagConfig, "config", "", configFlagHelp())
	pf.StringVarP(&flagOutputDir, "output-dir", "o", "", "directory for generated images (overrides config)")
	pf.BoolVar(&flagJSON, "json", false, "emit machine-readable JSON")
	pf.DurationVar(&flagTimeout, "timeout", 0, "HTTP request timeout (0 = client default; does not cap WS job duration)")
	rootCmd.SetFlagErrorFunc(func(cmd *cobra.Command, err error) error {
		return usageError{cmd: cmd, err: err}
	})
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
	// Hard cutover: a config left in the pre-consolidation directory is
	// reported with the exact move command rather than being read silently,
	// which would keep both locations alive indefinitely.
	if legacyErr := config.CheckLegacyLocation(flagConfig); legacyErr != nil {
		return nil, exitError{code: 2, err: legacyErr}
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
// explicit serverFlag > config file server_url. A missing value is an error
// that names every source, and a config that cannot be resolved or parsed is
// reported instead of being silently treated as "no URL".
func resolveServerURL(serverFlag, configFlag string) (string, error) {
	if raw := strings.TrimSpace(serverFlag); raw != "" {
		return normalizeServerURL(raw, "--server/$ST_SERVER")
	}

	path, err := config.Resolve(configFlag)
	if err != nil {
		return "", fmt.Errorf("resolve config path: %w\n  hint: pass --config, set $ST_CONFIG, or create the XDG default", err)
	}
	cfg, err := config.Load(path)
	if err != nil {
		return "", fmt.Errorf("load config %s: %w", path, err)
	}
	raw := strings.TrimSpace(cfg.ServerURL)
	if raw == "" {
		return "", fmt.Errorf(
			"no backend URL: set --server, $ST_SERVER, or server_url in %s\n"+
				"  example: export ST_SERVER=http://host:4200",
			path,
		)
	}
	return normalizeServerURL(raw, "server_url in "+path)
}

// normalizeServerURL makes a scheme-less host usable and rejects values that
// cannot become an HTTP endpoint, naming where the value came from.
func normalizeServerURL(raw, source string) (string, error) {
	normalized := raw
	if !strings.Contains(normalized, "://") {
		normalized = "http://" + normalized
	}
	u, err := url.Parse(normalized)
	if err != nil || u.Host == "" || (u.Scheme != "http" && u.Scheme != "https") {
		return "", fmt.Errorf(
			"invalid backend URL %q (from %s): expected http(s)://host[:port]",
			raw, source,
		)
	}
	return strings.TrimRight(u.String(), "/"), nil
}

// newClient builds an stclient pointed at the resolved server URL, honoring
// --timeout when set. URL precedence: --server/$ST_SERVER > config server_url.
func newClient(opts ...stclient.Option) (*stclient.Client, error) {
	baseURL, err := resolveServerURL(flagServer, flagConfig)
	if err != nil {
		return nil, err
	}
	if flagTimeout > 0 {
		opts = append(opts, stclient.WithHTTPClient(&http.Client{Timeout: flagTimeout}))
	}
	return stclient.New(baseURL, opts...), nil
}

// usageError marks an error as a command-usage problem so main can point at
// the command's help without duplicating the full usage text.
type usageError struct {
	cmd *cobra.Command
	err error
}

func (e usageError) Error() string { return e.err.Error() }
func (e usageError) Unwrap() error { return e.err }

func main() {
	if err := executeCLI(context.Background(), os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		var usage usageError
		if errors.As(err, &usage) && usage.cmd != nil {
			fmt.Fprintf(os.Stderr, "hint: run '%s --help'\n", usage.cmd.CommandPath())
		}
		os.Exit(exitCodeOf(err))
	}
}
