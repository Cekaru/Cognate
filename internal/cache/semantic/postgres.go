package semantic

// PostgresIndex is the persistent L2 backend: the same vector-search surface as
// MemoryIndex, stored in Postgres with the pgvector extension so the cache
// survives a restart (ROADMAP.md §6, the persistence step that finishes Phase
// 1). Cross-lingual matching is unchanged — BGE-M3 query vectors are stored in
// the prompt's original language and compared by cosine distance server-side.

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strconv"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/kaanrumin/polyglot-cache/internal/cache"
)

var _ Index = (*PostgresIndex)(nil)

// schemaDDL bootstraps the extension, table, and indexes. Every statement is
// idempotent so startup is safe against an already-provisioned database. The
// embedding column is fixed at the BGE-M3 dimensionality; a cosine HNSW index
// serves approximate nearest-neighbor search, and the (model, tenant_scope)
// b-tree supports the exact-match filters applied before the vector ordering.
var schemaDDL = []string{
	`CREATE EXTENSION IF NOT EXISTS vector`,
	fmt.Sprintf(`CREATE TABLE IF NOT EXISTS l2_entries (
		id           BIGSERIAL PRIMARY KEY,
		key          TEXT        NOT NULL,
		tenant_scope TEXT        NOT NULL,
		model        TEXT        NOT NULL,
		embedding    vector(%d)  NOT NULL,
		response     BYTEA       NOT NULL,
		lang         TEXT        NOT NULL,
		created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
	)`, cache.EmbeddingDim),
	`CREATE INDEX IF NOT EXISTS l2_entries_model_scope_idx ON l2_entries (model, tenant_scope)`,
	`CREATE INDEX IF NOT EXISTS l2_entries_embedding_idx ON l2_entries USING hnsw (embedding vector_cosine_ops)`,
}

// PostgresIndex is a pgvector-backed vector store. It is safe for concurrent use
// (the underlying pool is).
type PostgresIndex struct {
	pool   *pgxpool.Pool
	logger *slog.Logger
}

// NewPostgresIndex connects to databaseURL, verifies the connection, and applies
// the schema. It returns an error if any of those fail so the caller can decide
// whether to fall back to the in-memory backend.
func NewPostgresIndex(ctx context.Context, databaseURL string, logger *slog.Logger) (*PostgresIndex, error) {
	if logger == nil {
		logger = slog.Default()
	}
	pool, err := pgxpool.New(ctx, databaseURL)
	if err != nil {
		return nil, fmt.Errorf("connect pgvector: %w", err)
	}
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("ping pgvector: %w", err)
	}
	for _, stmt := range schemaDDL {
		if _, err := pool.Exec(ctx, stmt); err != nil {
			pool.Close()
			return nil, fmt.Errorf("apply schema: %w", err)
		}
	}
	logger.Info("pgvector L2 index ready", "embedding_dim", cache.EmbeddingDim)
	return &PostgresIndex{pool: pool, logger: logger}, nil
}

// Close releases the connection pool.
func (ix *PostgresIndex) Close() {
	ix.pool.Close()
}

// Add persists an entry. When CreatedAt is zero the database default (now())
// applies.
func (ix *PostgresIndex) Add(ctx context.Context, e *Entry) error {
	const q = `INSERT INTO l2_entries (key, tenant_scope, model, embedding, response, lang, created_at)
		VALUES ($1, $2, $3, $4::vector, $5, $6, $7)`
	createdAt := e.CreatedAt
	if createdAt.IsZero() {
		createdAt = time.Now()
	}
	if _, err := ix.pool.Exec(ctx, q, e.Key, e.TenantScope, e.Model, encodeVector(e.Embedding), e.Response, e.Lang, createdAt); err != nil {
		return fmt.Errorf("l2 add: %w", err)
	}
	return nil
}

// Search returns the single nearest entry by cosine similarity among rows
// matching model and one of scopes, or (nil, 0, nil) when none match. Similarity
// is 1 minus pgvector's cosine distance, so it lives in [-1, 1] exactly like the
// in-memory backend, and threshold gating stays the engine's job.
func (ix *PostgresIndex) Search(ctx context.Context, vec []float32, model string, scopes ...string) (*Entry, float64, error) {
	// A zero query vector has no direction; mirror the in-memory guard rather
	// than send an undefined cosine distance to the database.
	if isZeroVector(vec) {
		return nil, 0, nil
	}

	var (
		sb   strings.Builder
		args []any
	)
	args = append(args, encodeVector(vec), model)
	sb.WriteString(`SELECT key, tenant_scope, model, response, lang, created_at,
		1 - (embedding <=> $1::vector) AS similarity
		FROM l2_entries
		WHERE model = $2`)
	// Empty scopes means "all scopes" (parity with MemoryIndex).
	if len(scopes) > 0 {
		args = append(args, scopes) // pgx encodes []string as text[]
		sb.WriteString(` AND tenant_scope = ANY($3)`)
	}
	sb.WriteString(` ORDER BY embedding <=> $1::vector LIMIT 1`)

	var (
		e   Entry
		sim float64
	)
	err := ix.pool.QueryRow(ctx, sb.String(), args...).Scan(
		&e.Key, &e.TenantScope, &e.Model, &e.Response, &e.Lang, &e.CreatedAt, &sim,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, 0, nil
	}
	if err != nil {
		return nil, 0, fmt.Errorf("l2 search: %w", err)
	}
	return &e, sim, nil
}

// Len reports the number of stored entries. Intended for diagnostics; not part
// of the Index surface.
func (ix *PostgresIndex) Len(ctx context.Context) (int, error) {
	var n int
	if err := ix.pool.QueryRow(ctx, `SELECT count(*) FROM l2_entries`).Scan(&n); err != nil {
		return 0, fmt.Errorf("l2 len: %w", err)
	}
	return n, nil
}

// encodeVector renders a float32 slice as a pgvector text literal, e.g.
// "[0.1,0.2,0.3]", for casting with $n::vector.
func encodeVector(v []float32) string {
	var sb strings.Builder
	sb.WriteByte('[')
	for i, x := range v {
		if i > 0 {
			sb.WriteByte(',')
		}
		sb.WriteString(strconv.FormatFloat(float64(x), 'f', -1, 32))
	}
	sb.WriteByte(']')
	return sb.String()
}

func isZeroVector(v []float32) bool {
	for _, x := range v {
		if x != 0 {
			return false
		}
	}
	return true
}
