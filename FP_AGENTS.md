<!-- This file is managed by fp init. Do not edit by hand. -->

## FP Issue Tracking

This project uses **fp** for issue tracking. AI agents must follow these rules.

## Workflow Primers

Run `fp guide` for workflow primers and a snapshot of this project (registered statuses, properties, extensions).

```bash
fp guide              # List available primers
fp guide plan         # Planning workflow (alias: planning)
fp guide implement    # Implementation workflow
fp guide brainstorm   # Brainstorm authoring (alias: bs)
fp guide extension    # Extension authoring (aliases: extensions, ext)
```

Some primers bundle references reachable via `--references`, e.g. `fp guide brainstorm --references mermaid` for the Mermaid diagram guide.

### Task Tracking

- Use `fp issue` for all task tracking - do not use built-in todo tools
- Create subissues with `--parent` flag - never use markdown checklists (`- [ ]`)
- Break work into atomic tasks (1-3 hours each)

### Work Session Flow

1. `fp issue list --status todo` - find available work
2. `fp issue update --status in-progress <id>` - claim it before starting
3. `fp comment <id> "progress..."` - log at every milestone
4. `fp issue update --status done <id>` - mark complete when finished

When you commit work, mention the fp issue in the message, unless otherwise instructed.

### Progress Logging

- Run `fp comment <id> "..."` at every milestone
- Write comments after significant commits
- Always leave a final comment before ending session

### Commands Reference

```bash
fp tree [parent-id]        # View issue hierarchy (optionally only show tree of parent-id)
fp issue list --status X   # Filter by status (todo/in-progress/done)
fp search <query>          # Search issues (AND by default, OR, "phrases", -negation)
fp issue create --title "..." --parent X --property key=value
fp issue update --status X <id> --property key=value
fp comment <id> "message"
fp comment update <comment-id> "new message"  # Alias: edit
fp comment delete <comment-id>                # Soft-delete/hide a comment
fp context <id>            # Load full issue context
fp guide extension         # Print the extensions authoring guide
fp guide brainstorm        # Print the brainstorm-authoring guide
```

### Flag Conventions

- Standard attributes use dedicated flags: `--status`, `--priority`, `--parent`, `--title`, `--description`
- `fp issue update --depends "<ids>" <id>` replaces the entire dependency set. It does not append. To add one dependency, first inspect the current dependencies and pass the full desired set.
- `--property key=value` is for **extension-registered custom properties only** (e.g., labels, env, notes)
- Do not use `--property` for standard attributes — use the dedicated flags instead

### Extensions

FP is extensible via TypeScript extensions. Extensions can hook into issue and comment lifecycle events (e.g., before marking issues done, after comments are added) to automate workflows.

- Guide: `.fp/extensions/EXTENSIONS.md` (or run `fp guide extension` if the file is not present)
- Extensions live in `.fp/extensions/` as `.ts` files

### Brainstorms

Brainstorm plans (`fp brainstorm create` / `fp bs create`) support a rich markdown + Mermaid authoring surface. Before writing or editing one, load the bundled authoring docs:

- `fp guide brainstorm` — entrypoint; read this first
- `fp guide brainstorm --references mermaid` — diagram authoring reference

#### How to create a brainstorm from a populated issue

1. **Load context** — `fp context <id>` to read the issue description, clusters, and open questions
2. **Load the authoring guide** — `fp guide brainstorm` then `fp guide brainstorm --references mermaid`
3. **Write the markdown** to a temp file using brainstorm directives:
   - `:::callout{type="info|warning|decision|question"}` for key context, risks, decisions, open questions
   - `:::card{title="..."}` to group each idea cluster
   - `:label[text]{color="green|blue|orange|red|purple"}` for inline priority/status signals
   - Mermaid `flowchart TD` for architecture, `sequenceDiagram` for protocol flows
   - Tables for option comparisons within cards
4. **Create** — `fp brainstorm create /tmp/bs-output.md --title '<title>'`
5. **Iterate** — `fp bs show <id> --with-comments` to read feedback; `fp bs update <id> /tmp/revised.md` to revise

#### Brainstorm commands

```bash
fp brainstorm create <file.md> [--title "..."]       # Create from markdown file
fp brainstorm show <id> [--with-comments]             # Dump markdown (+ comments)
fp brainstorm update <id> <file.md>                   # Replace with revised markdown
fp brainstorm list                                    # List all brainstorms
fp brainstorm versions <id>                           # List version history
fp bs comments add <id> "comment text"               # Add a comment
fp bs comments resolve <id> <commentId>              # Resolve addressed feedback
fp bs comments delete <id> <commentId>               # Delete a comment
```

#### Directive vocabulary (quick ref)

```text
:::callout{type="info"}   — background context
:::callout{type="warning"} — risk or caveat
:::callout{type="decision"} — rationale for a choice
:::callout{type="question"} — open question needing resolution

:::card{title="Cluster name"}  — group related options

:label[High]{color="green"}    — priority / status inline
:label[Medium]{color="blue"}
:label[Low]{color="purple"}
:label[Blocked]{color="red"}
:label[In Progress]{color="orange"}
```

Mermaid rules: diagram type declaration must be the first non-empty line inside the fence. `flowchart TD` over `LR` for tall layouts. No `classDef` or inline `style` — renderer overrides them.
