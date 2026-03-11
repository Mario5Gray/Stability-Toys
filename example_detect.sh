#!/bin/bash
# Example usage of utils/model_detector.py (extensible detection system)

echo "========================================="
echo "Model Detection Examples"
echo "========================================="
echo ""

# Example 1: Detect single model
echo "Example 1: Detect single model"
echo "-------------------------------"
echo "Command: python -m utils.model_detector /models/sdxl-model.safetensors"
echo ""
# python -m utils.model_detector /models/sdxl-model.safetensors
echo ""

# Example 2: Detect LoRA
echo "Example 2: Detect LoRA type"
echo "-------------------------------"
echo "Command: python -m utils.model_detector /models/loras/anime.safetensors"
echo ""
# python -m utils.model_detector /models/loras/anime.safetensors
echo ""

# Example 3: JSON output
echo "Example 3: JSON output for automation"
echo "-------------------------------"
echo "Command: python -m utils.model_detector model.safetensors --json"
echo ""
# python -m utils.model_detector model.safetensors --json
echo ""

# Example 4: Pretty JSON
echo "Example 4: Pretty JSON output"
echo "-------------------------------"
echo "Command: python -m utils.model_detector model.safetensors --pretty"
echo ""
# python -m utils.model_detector model.safetensors --pretty
echo ""

# Example 5: Integration with server config
echo "Example 5: Auto-configure server based on detection"
echo "-------------------------------"
cat << 'EOF'
#!/bin/bash
MODEL_PATH="/models/my-model.safetensors"
RESULT=$(python -m utils.model_detector "$MODEL_PATH" --json)
CROSS_ATTN=$(echo "$RESULT" | jq -r '.cross_attention_dim')
VARIANT=$(echo "$RESULT" | jq -r '.variant')

if [ "$CROSS_ATTN" = "2048" ]; then
    echo "Detected SDXL model (variant: $VARIANT)"
    export MODEL_ROOT=/models
    export MODEL=$(basename "$MODEL_PATH")
    export DEFAULT_SIZE=1024x1024
elif [ "$CROSS_ATTN" = "768" ]; then
    echo "Detected SD1.5 model (variant: $VARIANT)"
    export MODEL_ROOT=/models
    export MODEL=$(basename "$MODEL_PATH")
    export DEFAULT_SIZE=512x512
fi

./runner.sh
EOF
echo ""

# Example 6: Organize models
echo "Example 6: Organize models by type"
echo "-------------------------------"
cat << 'EOF'
#!/bin/bash
# Scan and organize models

for model in /models/*.safetensors; do
    RESULT=$(python -m utils.model_detector "$model" --json)
    VARIANT=$(echo "$RESULT" | jq -r '.variant')
    CROSS_ATTN=$(echo "$RESULT" | jq -r '.cross_attention_dim')
    IS_LORA=$(echo "$RESULT" | jq -r '.is_lora')

    if [ "$IS_LORA" = "true" ]; then
        if [ "$CROSS_ATTN" = "768" ]; then
            DEST_DIR="/models/loras/sd15/"
        elif [ "$CROSS_ATTN" = "2048" ]; then
            DEST_DIR="/models/loras/sdxl/"
        else
            DEST_DIR="/models/loras/unknown/"
        fi
    else
        if [ "$CROSS_ATTN" = "768" ]; then
            DEST_DIR="/models/sd15/"
        elif [ "$CROSS_ATTN" = "2048" ]; then
            DEST_DIR="/models/sdxl/"
        else
            DEST_DIR="/models/unknown/"
        fi
    fi

    mkdir -p "$DEST_DIR"
    echo "Moving $model to $DEST_DIR"
    # mv "$model" "$DEST_DIR"
done
EOF
echo ""

# Example 7: Using custom detectors
echo "Example 7: Using custom detectors (LCM, Turbo, etc.)"
echo "-------------------------------"
cat << 'EOF'
#!/usr/bin/env python3
from utils.model_detector import ModelDetector
from utils.custom_detector_example import LCMDetector, TurboDetector

# Create detector with custom logic
detector = ModelDetector()
detector.add_detector(LCMDetector())
detector.add_detector(TurboDetector())

# Detect model
info = detector.detect("/models/sdxl-lcm.safetensors")

# Check metadata
if info.metadata.get("is_lcm"):
    print(f"LCM model detected! Use {info.metadata['recommended_steps']} steps")
    print(f"Recommended guidance: {info.metadata['recommended_guidance']}")

# Output as JSON
print(info.to_json(indent=2))
EOF
echo ""

echo "========================================="
echo "Try these commands with your models!"
echo "========================================="
echo ""
echo "For extensibility examples, see:"
echo "  - utils/custom_detector_example.py"
echo "  - docs/MODEL_DETECTOR_EXTENSIBLE.md"
