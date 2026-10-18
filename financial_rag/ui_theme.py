"""Context: shared presentation for the local financial research workspace."""


def apply_theme():
    """Context: style Streamlit's native controls without replacing their behavior."""
    import streamlit as st
    from urllib.parse import quote

    st.markdown("""
    <style>
    :root { --plum: #543043; --ink: #372631; --muted: #92838a; }
    .st-key-ratio-analysis, [class*="st-key-overview-chart-"] {
        background:#fcfbf9; padding:1rem 1.25rem !important; }
    .st-key-ratio-analysis [data-baseweb="tab-list"] { background:#f0eae9; border-radius:6px;
        gap:0; padding:3px; margin:.4rem 0 .8rem; }
    .st-key-ratio-analysis [data-baseweb="tab"] { flex:1; justify-content:center; border-radius:4px; font-size:.78rem; }
    .st-key-ratio-analysis [data-baseweb="tab"][aria-selected="true"] { background:#fcfbf9; box-shadow:0 1px 4px #5430430a; }
    .st-key-ratio-analysis [data-baseweb="tab-highlight"] { display:none; }
    .st-key-ratio-analysis [data-testid="stPopover"] button { border:0; background:transparent;
        padding:.1rem 0; color:#88717f; text-align:left; min-height:3rem; font-size:.8rem; }
    .ratio-card { border-right:1px solid #e5dce0; min-height:88px; padding:.2rem .6rem .7rem 0; }
    .st-key-ratio-analysis [data-testid="stColumn"]:last-child .ratio-card { border-right:0; }
    .ratio-value { font-size:2.1rem; line-height:1.6; font-variant-numeric:tabular-nums; }
    .ratio-comparison { font-size:.73rem; color:#8c6c7e; line-height:1.9; }
    .ratio-footer { border-top:1px solid #e5dce0; padding:1rem 0; margin-top:.6rem;
        font-size:.74rem; color:#927e89; }
    .source-viewer { padding: .3rem .5rem 1rem; }
    .source-eyebrow { font-size:.68rem; letter-spacing:.18em; color:#957a89; }
    .source-viewer h1 { margin:.5rem 0 1.3rem; font-size:2.4rem !important; }
    .source-badge { display:inline-block; background:#eee4e9; color:#64394d;
        padding:.45rem .75rem; border-radius:4px; font-size:.8rem; margin-bottom:1.5rem; }
    .source-excerpt { border-left:3px solid #b97b96; padding:.3rem 0 .3rem 1.5rem;
        font:1.5rem/1.7 Georgia,serif; overflow-wrap:anywhere; }
    .source-excerpt-text { white-space:pre-wrap; }
    .source-tabular { font: .9rem/1.8 monospace; overflow-x:auto; }
    .source-tabular .source-excerpt-text { white-space:pre; }
    .source-excerpt mark { background:#e5cfda; color:inherit; padding:.1rem 0; }
    .source-gap { font: .72rem/2 sans-serif; color:#927c88; margin:1rem 0; }
    .source-footer { border-top:1px solid #e5dce0; margin-top:1.7rem; padding-top:1rem;
        color:#927e89; font-size:.76rem; line-height:1.8; }
    .stApp { background: #f8f6f2; color: var(--ink); }
    [data-testid="stHeader"] { background: #f8f6f2e8; }
    .block-container { max-width: 1600px; padding: 1.4rem 2.2rem 3rem; }
    h1, h2, h3 { font-family: Georgia, 'Times New Roman', serif !important;
        color: var(--ink); font-weight: 400 !important; letter-spacing: -.035em; }
    h1 { font-size: clamp(2rem, 3.1vw, 3.1rem) !important; line-height: 1.15 !important; padding-bottom: .4rem !important; }
    h3 { font-size: 1.5rem !important; }
    p, label, button { font-size: .88rem; }
    [data-testid="stCaptionContainer"] { color: var(--muted); }
    [data-testid="stSidebar"] { background: #3e2533; border-right: 1px solid #4b303f;
        min-width: 250px; max-width: 270px; }
    [data-testid="stSidebarUserContent"] { padding: 1.3rem 1rem; }
    [data-testid="stSidebar"] * { color: #f4e9ee; }
    [data-testid="stSidebar"] [data-baseweb="select"] *,
    [data-testid="stSidebar"] input { color: #3e2533; }
    [data-testid="stSidebar"] [data-baseweb="tag"] * { color: #fff; }
    [data-testid="stSidebar"] [data-testid="stExpander"] { background: #482e3d; }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary p { font-size: .75rem; }
    /* Context: keep accessible radio navigation, styled as full-width menu items. */
    .st-key-workspace-navigation [role="radiogroup"] { gap: 6px !important; width: 100%; }
    .st-key-workspace-navigation label[data-baseweb="radio"] { display: flex !important;
        flex-direction: row !important; align-items: center !important; gap: 14px !important;
        width: 100%; min-height: 50px; margin: 0 !important; padding: 12px 14px !important;
        box-sizing: border-box; border-radius: 6px; cursor: pointer; transition: background .15s; }
    .st-key-workspace-navigation label[data-baseweb="radio"] > div:not([data-testid="stMarkdownContainer"]):not(:has([data-testid="stMarkdownContainer"])) { display: none !important; }
    .st-key-workspace-navigation input[type="radio"] { position: absolute; opacity: 0; }
    .st-key-workspace-navigation label[data-baseweb="radio"] p { font-size: .85rem;
        margin: 0; white-space: nowrap; line-height: 1.4; color: #e5d3dd; }
    .st-key-workspace-navigation label[data-baseweb="radio"]:hover { background: #513443; }
    .st-key-workspace-navigation label[data-baseweb="radio"]:has(input:checked) { background: #704859; }
    .st-key-workspace-navigation label[data-baseweb="radio"]:has(input:checked) p { color: #fff; }
    .st-key-workspace-navigation label[data-baseweb="radio"]:has(input:focus-visible) { outline: 2px solid #dcc2ce; outline-offset: 2px; }
    .st-key-workspace-navigation label[data-baseweb="radio"]:before { content: ''; display: block;
        width: 20px; height: 20px; flex: 0 0 20px; background: #dfcbd6;
        mask-repeat: no-repeat; mask-position: center; mask-size: contain; }
    [data-testid="stSidebar"] button[kind="secondary"] { background: transparent;
        border: 1px solid #735061; color: #f4e9ee; }
    [data-testid="stSidebar"] .workspace-label { font-size: .66rem; padding: .5rem 14px 1rem;
        letter-spacing: .16em; color: #c9acbc; white-space: nowrap; }
    [data-testid="stSidebar"] h2 { font-size: 1rem !important; }
    .workspace-brand { display: flex; align-items: center; font: 2.5rem Georgia, serif;
        letter-spacing: -.07em; padding: .2rem 0 2.4rem; color: #fff2f6; }
    .workspace-brand span { color: #cfb3c1; }
    .folio-mark { display: inline-flex; gap: 2px; margin-right: 10px; transform: skewY(-22deg); }
    .folio-mark i { display: block; width: 7px; height: 26px; background: #dec1cf; }
    .folio-mark i:nth-child(2) { opacity: .7; transform: translateY(-2px); }
    .folio-mark i:nth-child(3) { opacity: .4; transform: translateY(-4px); }
    .company-card { display: flex; gap: 10px; align-items: center; border: 1px solid #735061;
        border-radius: 6px; padding: 14px 10px; margin: 14px 0 8px; }
    .company-card b { font-size: .76rem; font-weight: 500; }
    .company-card small { display: block; font-size: .68rem; color: #cbb8c2 !important; margin-top: 5px; }
    .company-avatar { background: #765365; padding: 6px 10px; border-radius: 4px; font: 1.2rem Georgia, serif; }
    .workspace-label, .eyebrow { font-size: .63rem; text-transform: uppercase;
        letter-spacing: .18em; color: #a18594; padding: .5rem 0; }
    .workspace-note { border: 1px solid #765365; border-radius: 7px; background: #492f3d;
        padding: 14px; margin: 26px 0 0; font-size: .72rem; line-height: 1.8; }
    .breadcrumb { display: flex; justify-content: space-between; align-items: center; gap: 1rem;
        color: #998891; font-size: .73rem; padding: .1rem 0 1.2rem;
        border-bottom: 1px solid #e9e1e4; margin-bottom: 1.2rem; }
    .data-status { white-space: nowrap; font-size: .65rem; color: #998277; }
    .comparison-periods { font-size: .85rem; color: #583044; padding: 10px 0;
        font-variant-numeric: tabular-nums; border-bottom: 1px solid #e9e1e4; }
    .breadcrumb strong { color: #543043; font-weight: 500; }
    [data-baseweb="tab-list"] { gap: 1.5rem; background: transparent;
        border-bottom: 1px solid #e9e1e4; }
    [data-baseweb="tab"] { font-size: .8rem; padding-left: 0; padding-right: 0; }
    [data-testid="stMetric"] { padding: .25rem .8rem .4rem 0; border-right: 1px solid #e7dfe2; }
    [data-testid="stMetricLabel"] { color: #96838d; }
    [data-testid="stMetricValue"] { font-family: Georgia, serif; font-size: clamp(1.5rem, 2.4vw, 2.5rem); color: #392e35; }
    [data-testid="stMetricDelta"] { font-size: .72rem; }
    [data-testid="stVerticalBlockBorderWrapper"] > div { border-color: #e9e2e3 !important;
        border-radius: 8px !important; }
    [class*="st-key-overview-chart-"], [class*="st-key-statement-detail-"] { background: #fcfbf9; }
    [data-testid="stDataFrame"] { border-radius: 5px; }
    [class*="st-key-research-brief-panel-"] { background: #f0e9ec; border: 1px solid #e2d3db;
        border-radius: 7px; padding: 24px !important; gap: 0 !important; }
    .research-brief-header { padding-bottom: 22px; border-bottom: 1px solid #ddd0d7; }
    .research-brief-header h2 { font-size: 2rem !important; margin: 18px 0 12px;
        padding: 0; line-height: 1.2; color: #241d22; }
    .brief-label { display: flex; justify-content: space-between; align-items: center;
        gap: 6px; color: #926c80; text-transform: uppercase; letter-spacing: .13em; font-size: .55rem; }
    .brief-label span:last-child { border: 1px solid #d9c7d0; padding: 3px; border-radius: 2px;
        letter-spacing: .03em; font-size: .45rem; white-space: nowrap; }
    .research-brief-header .intro { color: #97818c; font-size: .77rem; line-height: 1.9; margin-bottom: 0; }
    .brief-item { border-bottom: 1px solid #ddd0d7; padding: 22px 0; }
    .brief-item h4 { font-family: sans-serif; font-weight: 600; font-size: .87rem;
        color: #281f25; margin: 8px 0 10px; line-height: 1.6; }
    .brief-item p { color: #836c78; font-size: .8rem; line-height: 1.95; margin-bottom: 0; }
    .brief-item .eyebrow { font-size: .6rem; color: #896679; letter-spacing: .12em; padding: 0; }
    [class*="st-key-research-brief-panel-"] [data-testid="stForm"] { border: 1px solid #d9c7d0;
        padding: 8px 10px; background: #fcfbf9; margin-top: 22px; border-radius: 6px; }
    [class*="st-key-research-brief-panel-"] [data-testid="stTextInput"] input { font-size: .75rem; }
    [class*="st-key-research-brief-panel-"] [data-testid="stCaptionContainer"] { text-align: center;
        font-size: .6rem; padding-top: 8px; }
    .statement-heading { display: flex; justify-content: space-between; align-items: center;
        padding-bottom: 16px; gap: 10px; font-size: .9rem; }
    .statement-heading span { color: #92838a; font-size: .65rem; font-family: monospace;
        text-align: right; margin-left: auto; }
    .reporting-unit { color: #92838a; font: .65rem monospace; text-align: right;
        font-variant-numeric: tabular-nums; letter-spacing: .01em; }
    .metric-reporting-unit { padding-right: .8rem; margin-bottom: -8px; }
    .st-key-fiscal-year-picker button { background: #efe8eb; border: 1px solid #704859;
        color: #543043; min-height: 42px; border-radius: 8px; }
    .statement-tab { width: fit-content; border-bottom: 2px solid #704859;
        padding-bottom: 10px; color: #704859; font-size: .72rem; }
    .statement-scroll { overflow-x: auto; }
    .statement-table { width: 100%; border-collapse: collapse; font-size: .73rem; }
    .statement-table th { text-align: right; font-size: .62rem; font-weight: 400; color: #92838a;
        padding: 12px 10px; border-bottom: 1px solid #e9e1e4; white-space: nowrap; }
    .statement-table td { text-align: right; padding: 12px 10px; border-bottom: 1px solid #f0e9ec;
        font-variant-numeric: tabular-nums; white-space: nowrap; }
    .statement-table th:first-child, .statement-table td:first-child { text-align: left; white-space: normal; min-width: 125px; }
    .statement-total { background: #f4eef1; color: #583044; }
    .analysis-table-wrap { overflow-x: auto; }
    .analysis-table { width: 100%; border-collapse: collapse; font-size: .76rem; }
    .analysis-table th { color: #92838a; font-weight: 400; font-size: .68rem; text-align: right; }
    .analysis-table th, .analysis-table td { padding: 12px 8px; border-bottom: 1px solid #e9e1e4; vertical-align: top; }
    .analysis-table td { text-align: right; font-variant-numeric: tabular-nums; }
    .analysis-table th:first-child, .analysis-table td:first-child { text-align: left; width: 48%; }
    .metric-formula summary { cursor: pointer; list-style: none; border-radius: 3px; }
    .metric-formula summary::-webkit-details-marker { display: none; }
    .formula-action { display: block; color: #704859; font-size: .65rem; opacity: 0; margin-top: 4px; }
    .metric-formula:hover .formula-action, .metric-formula:focus-within .formula-action,
    .metric-formula[open] .formula-action { opacity: 1; text-decoration: underline; }
    .metric-formula summary:focus-visible { outline: 2px solid #704859; outline-offset: 3px; }
    .formula-content { background: #f2eaf0; padding: 12px; border-radius: 5px;
        margin-top: 9px; font-size: .7rem; line-height: 1.8; overflow-wrap: anywhere; }
    .formula-content p { font-size: .7rem; margin: 9px 0 0; }
    @media (hover: none) { .formula-action { opacity: 1; } }
    .statement-section td { padding-top: 20px; color: #8e7481; font-size: .7rem;
        font-weight: 600; text-align: left; background: #fcfbf9; }
    .statement-detail td:first-child { padding-left: 24px; }
    .statement-margin td { color: #8e7481; font-size: .68rem; font-style: italic;
        padding-top: 6px; padding-bottom: 10px; background: #fcfbf9; }
    .statement-margin td:first-child { padding-left: 24px; }
    .statement-section-label { margin-top: 20px; padding: 8px 10px; color: #8e7481;
        font-size: .7rem; font-weight: 600; }
    button[kind="primary"] { background: #583044; border-color: #583044; }
    .st-key-question-actions { display: flex; flex-direction: row !important;
        flex-wrap: wrap; gap: 10px !important; align-items: center; }
    .st-key-question-actions > div { width: auto !important; flex: 0 0 auto !important; }
    [role="dialog"] { background: #fcfbf9; border: 1px solid #e5dce0; border-radius: 12px;
        box-shadow: 0 24px 90px #35232f33; }
    .welcome-overview { padding: 0 12px 8px; }
    .welcome-overview h1 { font: 2.6rem Georgia, serif; margin: 0 0 22px; }
    .welcome-intro { color: #8b7e85; line-height: 1.8; margin-bottom: 18px; }
    .welcome-step { display: flex; gap: 16px; margin-bottom: 22px; }
    .welcome-step > span { color: #704859; font: .85rem monospace; padding-top: 3px; }
    .welcome-step b { font-size: .9rem; font-weight: 600; color: #392e35; }
    .welcome-step p { color: #8b7e85; line-height: 1.8; margin: 5px 0 0; }
    .welcome-footer { border-top: 1px solid #e5dce0; padding-top: 17px; color: #8b7e85;
        font-size: .75rem; line-height: 1.9; }
    @media (max-width: 800px) {
        .block-container { padding: 1rem; }
        [data-testid="stMetricValue"] { font-size: 1.55rem; }
        [class*="st-key-research-brief-panel-"] { padding: 16px !important; }
    }
    </style>
    """, unsafe_allow_html=True)
    # Context: inline vector masks provide consistent line icons without font or image downloads.
    icons=(
        '<path d="M4 3v17h17M8 7v9m5-12v12m5-7v7"/>',
        '<path d="m12 3 9 5-9 5-9-5 9-5Zm-9 9 9 5 9-5M3 16l9 5 9-5"/>',
        '<path d="M4 4v16m5-14v14m4-16 5 16m1-16 3 15"/>',
        '<path d="M6 21V4a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v17l-6-4-6 4Z"/>',
        '<circle cx="10" cy="10" r="6"/><path d="m15 15 6 6"/>',
        '<circle cx="12" cy="12" r="9"/><path d="m16 8-3 5-5 3 3-5 5-3Z"/>',
        '<circle cx="12" cy="12" r="9"/><path d="M12 11v6m0-11v1"/>',
    )
    rules=[]
    for number,paths in enumerate(icons,1):
        svg='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">'+paths+'</svg>'
        rules.append('.st-key-workspace-navigation label[data-baseweb="radio"]:nth-of-type('
                     +str(number)+'):before { mask-image: url("data:image/svg+xml,'+quote(svg,safe='')+'"); }')
    st.markdown('<style>'+''.join(rules)+'</style>',unsafe_allow_html=True)
