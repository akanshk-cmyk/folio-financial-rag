# Context: run from the project root after configuring SEC_USER_AGENT in .env.
# Downloads one verified primary filing, preserves its original HTML, and then
# appends canonical document records. It does not invoke paid generation APIs.
$ErrorActionPreference = 'Stop'
$sourceManifest = Get-Content -LiteralPath 'examples/apple_2025_source.json' -Raw | ConvertFrom-Json

# Context: use the locked environment; the downloader validates the contact header.
uv run --locked financial-rag download $sourceManifest.primary_document_url --output $sourceManifest.local_path
if ($LASTEXITCODE -ne 0) {
    throw 'Download failed; ingestion was not attempted.'
}

# Context: verified fiscal and filing identifiers remain distinct in source metadata.
uv run --locked financial-rag ingest $sourceManifest.local_path --ticker $sourceManifest.ticker --year $sourceManifest.fiscal_year --quarter $sourceManifest.fiscal_quarter --doc-type $sourceManifest.doc_type --source $sourceManifest.primary_document_url --accession $sourceManifest.accession --filing-date $sourceManifest.filing_date --output data/documents.json
if ($LASTEXITCODE -ne 0) {
    throw 'Ingestion failed; inspect the preserved primary HTML before retrying.'
}

# Context: build a free local keyword index first to inspect parsing before models.
uv run --locked financial-rag index --strategy enhanced --backend lexical --output data/apple-lexical.json
if ($LASTEXITCODE -ne 0) {
    throw 'Local indexing failed.'
}
