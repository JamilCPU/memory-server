#!/bin/bash
# Run this on each Raspberry Pi to configure memory MCP access.
# Usage: bash pi_setup.sh

MEMORY_SERVER="${MEMORY_SERVER:-http://<memory-server-ip>:8765/sse}"
SETTINGS_FILE="$HOME/.claude/settings.json"

mkdir -p "$HOME/.claude"

# If settings.json doesn't exist, create it
if [ ! -f "$SETTINGS_FILE" ]; then
    echo '{}' > "$SETTINGS_FILE"
fi

# Check if python3 and jq are available
if ! command -v jq &> /dev/null; then
    echo "Installing jq..."
    sudo apt-get install -y jq
fi

# Inject mcpServers block
UPDATED=$(jq --arg url "$MEMORY_SERVER" '
  .mcpServers["memory"] = {"url": $url}
' "$SETTINGS_FILE")

echo "$UPDATED" > "$SETTINGS_FILE"

echo "Done. MCP memory server configured at $MEMORY_SERVER"
echo "Settings written to $SETTINGS_FILE"
echo ""
echo "Verify with: cat $SETTINGS_FILE"
echo ""
echo "Also copy the global CLAUDE.md from the PC to $HOME/.claude/CLAUDE.md"
echo "  scp user@192.0.2.99:/mnt/f/Dev/memory-server/CLAUDE.md $HOME/.claude/CLAUDE.md"
