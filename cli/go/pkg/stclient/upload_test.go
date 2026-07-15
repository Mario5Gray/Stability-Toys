package stclient

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestUploadReturnsFileRef(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/upload" || r.Method != http.MethodPost {
			t.Fatalf("unexpected %s %s", r.Method, r.URL.Path)
		}
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Fatal(err)
		}
		if _, _, err := r.FormFile("file"); err != nil {
			t.Fatalf("no file part: %v", err)
		}
		w.Write([]byte(`{"fileRef":"abc123"}`))
	}))
	defer srv.Close()

	ref, err := New(srv.URL).Upload(context.Background(), "x.png", []byte("PNGBYTES"), "")
	if err != nil {
		t.Fatal(err)
	}
	if ref != "abc123" {
		t.Fatalf("got %q", ref)
	}
}

func TestUploadSendsBucketFormField(t *testing.T) {
	var gotType string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Fatal(err)
		}
		gotType = r.FormValue("type")
		w.Write([]byte(`{"fileRef":"R-abc"}`))
	}))
	defer srv.Close()

	ref, err := New(srv.URL).Upload(context.Background(), "map.png", []byte("data"), "canny")
	if err != nil {
		t.Fatal(err)
	}
	if ref != "R-abc" {
		t.Errorf("ref = %q, want R-abc", ref)
	}
	if gotType != "canny" {
		t.Errorf("form type = %q, want canny", gotType)
	}
}

func TestUploadNoTypeFieldWhenBucketEmpty(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		r.ParseMultipartForm(1 << 20)
		if v := r.FormValue("type"); v != "" {
			t.Errorf("expected no type field when bucket is empty, got %q", v)
		}
		w.Write([]byte(`{"fileRef":"R-xyz"}`))
	}))
	defer srv.Close()

	New(srv.URL).Upload(context.Background(), "img.png", []byte("data"), "")
}

func TestSuperResSendsFileAndMagnitudeReturnsImage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/superres" || r.Method != http.MethodPost {
			t.Fatalf("unexpected %s %s", r.Method, r.URL.Path)
		}
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Fatal(err)
		}
		if _, _, err := r.FormFile("file"); err != nil {
			t.Fatalf("no file part: %v", err)
		}
		if got := r.FormValue("magnitude"); got != "3" {
			t.Fatalf("magnitude = %q, want 3", got)
		}
		w.Header().Set("Content-Type", "image/png")
		w.Write([]byte("SRIMAGE"))
	}))
	defer srv.Close()

	out, err := New(srv.URL).SuperRes(context.Background(), []byte("INPUT"), 3)
	if err != nil {
		t.Fatal(err)
	}
	if string(out) != "SRIMAGE" {
		t.Fatalf("got %q, want SRIMAGE", out)
	}
}

func TestFetchStorageReturnsBytes(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/storage/out-key.png" || r.Method != http.MethodGet {
			t.Fatalf("unexpected %s %s", r.Method, r.URL.Path)
		}
		w.Write([]byte("RESULTPNG"))
	}))
	defer srv.Close()

	out, err := New(srv.URL).FetchStorage(context.Background(), "out-key.png")
	if err != nil {
		t.Fatal(err)
	}
	if string(out) != "RESULTPNG" {
		t.Fatalf("got %q", out)
	}
}

func TestFetchStorageErrorsOn404(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.NotFound(w, r)
	}))
	defer srv.Close()

	if _, err := New(srv.URL).FetchStorage(context.Background(), "missing"); err == nil {
		t.Fatal("expected error on 404, got nil")
	}
}

func TestUploadFileReturnsResolvedBucketAndDims(t *testing.T) {
	var gotType string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = r.ParseMultipartForm(1 << 20)
		gotType = r.FormValue("type")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"fileRef":"R1","bucket":"control_map","width":8,"height":6}`))
	}))
	defer srv.Close()

	res, err := New(srv.URL).UploadFile(context.Background(), "m.png", []byte("data"), "canny")
	if err != nil {
		t.Fatal(err)
	}
	if gotType != "canny" {
		t.Fatalf("type field = %q", gotType)
	}
	if res.Ref != "R1" || res.Bucket != "control_map" || res.Width != 8 || res.Height != 6 {
		t.Fatalf("bad result: %+v", res)
	}
}

func TestUploadDelegatesAndReturnsRef(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"fileRef":"R2","bucket":"upload"}`))
	}))
	defer srv.Close()

	ref, err := New(srv.URL).Upload(context.Background(), "x.png", []byte("data"), "")
	if err != nil {
		t.Fatal(err)
	}
	if ref != "R2" {
		t.Fatalf("ref = %q", ref)
	}
}
