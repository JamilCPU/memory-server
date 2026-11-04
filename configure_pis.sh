#!/bin/bash
# Configures OpenClaw (Claude Code) on Raspberry Pis to use the shared memory server.
# Usage: bash configure_pis.sh <ssh-user> <ssh-password>
#
# Requires: sshpass  (apt install sshpass)

SSH_USER="${1:-pi}"
SSH_PASS="${2:-}"
MEMORY_URL="${MEMORY_URL:-http://<memory-server-ip>:8765/sse}"
read -r -a PI_IPS <<< "${PI_IPS:-<pi-ip-1> <pi-ip-2> <pi-ip-3>}"

if [ -z "$SSH_PASS" ]; then
    echo "Usage: bash configure_pis.sh <user> <password>"
    exit 1
fi

MCP_JSON='{
  "mcpServers": {
    "memory": {
      "url": "'"$MEMORY_URL"'"
    }
  }
}'

CLAUDE_MD='# Global Agent Instructions

## Shared Memory

You have access to a shared memory MCP server (tool prefix: `memory`) that stores problem/solution pairs across all agents and devices on this network.

### When to search memory
Call `search_memory` before starting any task that:
- Involves system configuration, package installation, or dependency issues
- Requires networking, hardware, or OS-level setup
- Feels unfamiliar or has uncertainty about the correct approach
- Previously caused errors in this session

### When to save to memory
Call `save_memory` after:
- Solving a problem that required debugging, trial and error, or research
- Finding a non-obvious fix (a flag, a workaround, a specific command sequence)
- Resolving a recurring issue for the first time
- Successfully completing a multi-step setup that was not straightforward

Do NOT save trivial operations (standard CRUD, basic file edits, simple installs that worked first try).

### Format guidance
When saving:
- `problem`: be specific and searchable — include error messages, package names, or symptoms
- `solution`: include the exact commands or code that worked, not just a description
- `tags`: use lowercase keywords like `["apt", "opencv", "i2c", "networking", "python"]`
- `device`: use the hostname of the machine where you solved it
'

for IP in "${PI_IPS[@]}"; do
    echo ""
    echo "=== Configuring $IP ==="

    # Test connectivity first
    if ! sshpass -p "$SSH_PASS" ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$SSH_USER@$IP" "echo ok" &>/dev/null; then
        echo "  SKIP: Cannot reach $IP"
        continue
    fi

    sshpass -p "$SSH_PASS" ssh -o StrictHostKeyChecking=no "$SSH_USER@$IP" bash <<EOF
        mkdir -p ~/.claude
        cat > ~/.claude/mcp.json << 'MCPEOF'
$MCP_JSON
MCPEOF
        cat > ~/.claude/CLAUDE.md << 'MDEOF'
$CLAUDE_MD
MDEOF
        echo "  Written ~/.claude/mcp.json"
        echo "  Written ~/.claude/CLAUDE.md"
        hostname
EOF

    echo "  Done: $IP"
done

echo ""
echo "All Pis configured. Verify with:"
echo "  sshpass -p <pass> ssh <user>@192.0.2.101 'cat ~/.claude/mcp.json'"
