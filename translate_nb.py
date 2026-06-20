import json

nb = json.load(open('experimental_results.ipynb', encoding='utf-8'))

replacements = [
    # Cell 20 - t1_load
    (20, '# segundo de inicio da janela de analise', '# start of analysis window (seconds)'),
    (20, '# segundo de fim da janela de analise', '# end of analysis window (seconds)'),
    (20, '# 1. Moving average nos dados brutos', '# 1. Moving average on raw data'),
    (20, '# 2. Mascara da janela de analise (para calibracao e plots)', '# 2. Analysis window mask'),
    (20, 'amostras na janela', 'samples in window'),
    (20, '# 3. Calibracao: min/max calculados SO dentro da janela', '# 3. Calibration: min/max computed only within the window'),
    (20, '# Normalizar todo o sinal com os limites da janela (pode sair de [-1,1] fora)', '# Normalize full signal with window limits (may exceed [-1,1] outside window)'),
    (20, '# 4. Media ponderada pelo span (dentro da janela)', '# 4. Span-weighted mean (within window)'),
    (20, '  apertura na janela', '  aperture in window'),
    # Cell 22 - t1_calib_plot
    (22, 'Retas de calibracao por canal e por teste  ', 'Calibration lines per channel and per test  '),
    (22, '(min/max calculado na janela', '(min/max computed in window'),
    (22, '# Reta de calibracao linear', '# Linear calibration line'),
    (22, '# Pontos de referencia', '# Reference points'),
    (22, '# Histograma so dentro da janela de analise', '# Histogram within analysis window only'),
    # Cell 24 - t1_signal
    (24, 'Apertura media dos 5 canais — janela de analise', 'Mean aperture of 5 channels — analysis window'),
    (24, '(−1 = fechada  +1 = aberta)', '(−1 = closed  +1 = open)'),
    (24, '# Canais individuais', '# Individual channels'),
    (24, '# Media em destaque', '# Mean (highlighted)'),
    (24, '"Media (5 canais)"', '"Mean (5 channels)"'),
    (24, 'set_xlabel("Tempo (s)")', 'set_xlabel("Time (s)")'),
    # Cell 26 - t2_load
    (26, '# ajustar se necessario', '# adjust if needed'),
    (26, '# 1. Moving average nos dados brutos', '# 1. Moving average on raw data'),
    (26, '# 2. Janela de analise', '# 2. Analysis window'),
    (26, 'amostras na janela', 'samples in window'),
    (26, '# 3. Calibracao: min/max dentro da janela', '# 3. Calibration: min/max within window'),
    (26, '"apertura [', '"aperture ['),
    # Cell 28 - t2_calib_plot
    (28, 'TEST 2 — Retas de calibracao  (min/max na janela', 'TEST 2 — Calibration lines  (min/max in window'),
    (28, '# Histograma dos dados da janela', '# Histogram of window data'),
    # Cell 30 - t2_signal
    (30, '# Ordem anatomica', '# Anatomical order'),
    (30, 'Apertura por dedo  (janela', 'Aperture per finger  (window'),
    (30, 'descida = dedo fechado  |  subida = mao aberta', 'drop = finger closed  |  rise = hand open'),
    (30, 'label="fechado"', 'label="closed"'),
    (30, '# Anotacao do minimo', '# Minimum annotation'),
    (30, 'set_xlabel("Tempo (s)", fontsize=10)', 'set_xlabel("Time (s)", fontsize=10)'),
    # Cell 33 - t6_plot
    (33, 'Apertura media por experiencia', 'Mean aperture per trial'),
    (33, 'set_xlabel("Tempo (s)")', 'set_xlabel("Time (s)")'),
    (33, 'Aperture  (-1=fechada  +1=aberta)', 'Aperture  (-1=closed  +1=open)'),
    # Cell 34 - t6_drift
    (34, '# ── carregar e calcular apertura media de cada trial ──', '# ── load and compute mean aperture per trial ──'),
    (34, '# pontos na grelha normalizada [0,1]', '# points on normalized grid [0,1]'),
    (34, '# ── grelha temporal normalizada ──', '# ── normalized time grid ──'),
    (34, '# normalizar para [0,1]', '# normalize to [0,1]'),
    (34, '# ── drift linear por trial ──', '# ── linear drift per trial ──'),
    (34, 'apertura/min', 'aperture/min'),
    (34, 'inicio=', 'start='),
    (34, 'fim=', 'end='),
    (34, '# ── histerese: separar fases ascendentes vs descendentes ──', '# ── hysteresis: separate ascending vs descending phases ──'),
    (34, 'Analise de drift, consistencia e histerese', 'Drift, consistency and hysteresis analysis'),
    (34, '# ── Plot 1: consistencia entre trials (tempo normalizado) ──', '# ── Plot 1: consistency across trials (normalized time) ──'),
    (34, 'Tempo normalizado (0=inicio  1=fim do trial)', 'Normalized time (0=start  1=end of trial)'),
    (34, 'label="Media"', 'label="Mean"'),
    (34, 'Consistencia entre trials (tempo normalizado)', 'Consistency across trials (normalized time)'),
    (34, '# ── Plot 2: drift linear por trial ──', '# ── Plot 2: linear drift per trial ──'),
    (34, 'set_xlabel("Tempo (s)")', 'set_xlabel("Time (s)")'),
    (34, 'Drift linear (reta de tendencia por trial)', 'Linear drift (trend line per trial)'),
    # Cell 36 - t7_plot
    (36, 'Apertura media por experiencia', 'Mean aperture per trial'),
    (36, 'set_xlabel("Tempo (s)")', 'set_xlabel("Time (s)")'),
    (36, 'Aperture  (-1=fechada  +1=aberta)', 'Aperture  (-1=closed  +1=open)'),
    # Cell 37 - t7_drift
    (37, '# ── carregar e calcular apertura media de cada trial ──', '# ── load and compute mean aperture per trial ──'),
    (37, '# pontos na grelha normalizada [0,1]', '# points on normalized grid [0,1]'),
    (37, '# ── grelha temporal normalizada ──', '# ── normalized time grid ──'),
    (37, '# normalizar para [0,1]', '# normalize to [0,1]'),
    (37, '# ── drift linear por trial ──', '# ── linear drift per trial ──'),
    (37, 'apertura/min', 'aperture/min'),
    (37, 'inicio=', 'start='),
    (37, 'fim=', 'end='),
    (37, '# ── histerese: separar fases ascendentes vs descendentes ──', '# ── hysteresis: separate ascending vs descending phases ──'),
    (37, 'Analise de drift, consistencia e histerese', 'Drift, consistency and hysteresis analysis'),
    (37, '# ── Plot 1: consistencia entre trials (tempo normalizado) ──', '# ── Plot 1: consistency across trials (normalized time) ──'),
    (37, 'Tempo normalizado (0=inicio  1=fim do trial)', 'Normalized time (0=start  1=end of trial)'),
    (37, 'label="Media"', 'label="Mean"'),
    (37, 'Consistencia entre trials (tempo normalizado)', 'Consistency across trials (normalized time)'),
    (37, '# ── Plot 2: drift linear por trial ──', '# ── Plot 2: linear drift per trial ──'),
    (37, 'set_xlabel("Tempo (s)")', 'set_xlabel("Time (s)")'),
    (37, 'Drift linear (reta de tendencia por trial)', 'Linear drift (trend line per trial)'),
    # Cell 39 - t8_analysis
    (39, '# remover DC', '# remove DC'),
    (39, '# FFT e soma dos 3 eixos', '# FFT and sum of 3 axes'),
    (39, '# Banda de tremor', '# Tremor band'),
    (39, '# % tempo com tremor (janela deslizante)', '# % time with tremor (sliding window)'),
    (39, '# ── carregar e analisar todos os ficheiros ──', '# ── load and analyse all files ──'),
    (39, 'banda 4-8 Hz a vermelho', '4-8 Hz band in red'),
    (39, 'Resumo da detecao de tremores (vermelho = possivel tremor >20%)', 'Tremor detection summary (red = possible tremor >20%)'),
    (39, 'label="limiar 20%"', 'label="threshold 20%"'),
    (39, 'set_ylabel("Potencia relativa 4-8 Hz (%)")', 'set_ylabel("Relative power 4-8 Hz (%)")'),
    (39, 'set_title("Potencia relativa na banda de tremor")', 'set_title("Relative power in tremor band")'),
    (39, 'label="banda 4-8 Hz"', 'label="4-8 Hz band"'),
    (39, 'set_ylabel("Frequencia dominante (Hz)")', 'set_ylabel("Dominant frequency (Hz)")'),
    (39, 'set_title("Frequencia dominante na banda 4-8 Hz")', 'set_title("Dominant frequency in 4-8 Hz band")'),
    (39, 'set_ylabel("% tempo com tremor activo")', 'set_ylabel("% time with active tremor")'),
    (39, 'set_title("Percentagem de tempo com tremor detectado")', 'set_title("Percentage of time with detected tremor")'),
    # Cell 43 - t8_full_spectra
    (43, 'pico_global=', 'global_peak='),
    (43, 'pico_4-8Hz=', 'band_peak_4-8Hz='),
    (43, 'Espectro completo (0-30 Hz) | verde=pico global | laranja=pico na banda 4-8Hz', 'Full spectrum (0-30 Hz) | green=global peak | orange=band peak 4-8Hz'),
    (43, 'label="banda 4-8 Hz"', 'label="4-8 Hz band"'),
    (43, '"pico global: ', '"global peak: '),
    (43, '"pico 4-8Hz: ', '"band peak: '),
    (43, 'set_xlabel("Frequencia (Hz)"', 'set_xlabel("Frequency (Hz)"'),
    (43, 'pico global=', 'global peak='),
    # Cell 47 - t9_load
    (47, 'Le CSV do jogo: salta header de calibracao, devolve df e spans.', 'Read game CSV: skip calibration header, return df and spans.'),
    (47, '# suavizacao ligeira (dados ja normalizados pelo jogo)', '# light smoothing (data already normalized by game)'),
    (47, '# suavizacao ligeira', '# light smoothing'),
    (47, '# usar spans da calibracao como pesos (ja existem no ficheiro)', '# use calibration spans as weights (stored in file header)'),
    (47, '# garantir que nenhum peso e zero', '# ensure no weight is zero'),
    (47, 'amostras na janela', 'samples in window'),
    (47, '  apertura na janela', '  aperture in window'),
    # Cell 48 - t9_plot
    (48, 'Apertura media', 'Mean aperture'),
    (48, '(-1=fechada  +1=aberta)', '(-1=closed  +1=open)'),
    (48, 'label="Media (5 dedos)"', 'label="Mean (5 fingers)"'),
    (48, 'set_xlabel("Tempo (s)")', 'set_xlabel("Time (s)")'),
]

not_found = []
for idx, old, new in replacements:
    src = ''.join(nb['cells'][idx]['source'])
    if old in src:
        src = src.replace(old, new)
        nb['cells'][idx]['source'] = src.splitlines(keepends=True)
        nb['cells'][idx]['outputs'] = []
        nb['cells'][idx]['execution_count'] = None
    else:
        not_found.append((idx, old[:50]))

with open('experimental_results.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

if not_found:
    print('NOT FOUND:')
    for idx, s in not_found:
        print(f'  cell {idx}: {s}')
else:
    print('All replacements applied successfully')
