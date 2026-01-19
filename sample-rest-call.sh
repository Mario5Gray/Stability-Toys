curl -X POST http://localhost:4200/generate   -H "Content-Type: application/json"   -o out.png   -d '{
    "prompt": "a cinematic photograph of a futuristic city at sunset",
    "size": "512x512",
    "num_inference_steps": 4,
    "guidance_scale": 1.0,
    "seed": 2861337
    "superres": true,
    "superres_format": "png"
  }'
