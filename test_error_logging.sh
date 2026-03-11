#!/bin/bash
# Test script to verify error logging is working

echo "Testing error logging in FastAPI server..."
echo ""

# Test 1: Invalid endpoint (404)
echo "Test 1: Invalid endpoint (should return 404)"
curl -s -X GET http://localhost:4200/invalid-endpoint || true
echo ""
echo ""

# Test 2: Invalid request body (422 validation error)
echo "Test 2: Invalid request body (should return 422)"
curl -s -X POST http://localhost:4200/generate \
  -H "Content-Type: application/json" \
  -d '{"invalid": "data"}' || true
echo ""
echo ""

# Test 3: Valid request (should succeed)
echo "Test 3: Valid request to /health (should return 200)"
curl -s -X GET http://localhost:4200/health
echo ""
echo ""

# Test 4: Generate with invalid size
echo "Test 4: Invalid size format (should return 422)"
curl -s -X POST http://localhost:4200/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "test", "size": "invalid"}' || true
echo ""
echo ""

echo "Done! Check docker logs to see if errors are being logged properly."
echo "Run: docker logs <container-name> 2>&1 | tail -50"
