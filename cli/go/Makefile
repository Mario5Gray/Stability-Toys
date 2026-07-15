# openapi.snapshot.json is OpenAPI 3.1 (verbatim from the FastAPI backend, used
# by the gated drift guard). oapi-codegen only supports 3.0.x, so downspec emits
# a throwaway 3.0 intermediate that codegen consumes.
gen:
	go run ./tools/downspec openapi.snapshot.json openapi.codegen.json
	go run github.com/oapi-codegen/oapi-codegen/v2/cmd/oapi-codegen -config oapi-codegen.yaml openapi.codegen.json
	rm -f openapi.codegen.json

build:
	go build ./...

install:
	go install ./cmd/st ./cmd/stcn

test:
	go vet ./...
	go test ./...
