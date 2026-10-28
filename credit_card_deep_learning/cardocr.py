"""OCR de cartões de crédito: pré-processamento, features, modelos clássicos (OpenCV ML)
e modelo pré-treinado (EasyOCR / PyTorch)."""
import glob
import os
import re

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

DIGIT_W, DIGIT_H = 32, 48


# ---------------------------------------------------------------------------
# Carregamento dos dados
# ---------------------------------------------------------------------------
def load_ds1(root):
    """Dataset 1: um dígito por imagem, pastas 0..9."""
    X, y = [], []
    for d in range(10):
        for p in sorted(glob.glob(os.path.join(root, str(d), '*'))):
            img = cv2.imread(p)
            if img is not None:
                X.append(img)
                y.append(d)
    return X, np.array(y)


def load_ds2_groups(folder):
    """Dataset 2: recortes 120x46 com 4 caracteres; rótulo nos 4 primeiros chars do nome
    ('_' = posição sem dígito)."""
    imgs, labels = [], []
    for p in sorted(glob.glob(os.path.join(folder, '*.png'))):
        imgs.append(cv2.imread(p))
        labels.append(os.path.basename(p)[:4])
    return imgs, labels


def split_group(img, n=4):
    """Divide um recorte de grupo em n fatias de largura igual (um dígito por fatia)."""
    w = img.shape[1]
    step = w / n
    return [img[:, int(round(i * step)):int(round((i + 1) * step))] for i in range(n)]


# ---------------------------------------------------------------------------
# Pré-processamento e extração de features
# ---------------------------------------------------------------------------
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
_hog = cv2.HOGDescriptor((DIGIT_W, DIGIT_H), (16, 16), (8, 8), (8, 8), 9)


def preprocess_digit(img):
    """Cinza + CLAHE + redimensionamento para 32x48."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = cv2.resize(g, (DIGIT_W, DIGIT_H), interpolation=cv2.INTER_AREA)
    return _clahe.apply(g)


def feat_raw(img):
    g = cv2.resize(preprocess_digit(img), (16, 24), interpolation=cv2.INTER_AREA)
    return (g.astype(np.float32) / 255.0).ravel()


def feat_hog(img):
    return _hog.compute(preprocess_digit(img)).ravel()


FEATURES = {'raw': feat_raw, 'hog': feat_hog}


def augment(img, rng, n=3):
    """Pequenas translações/escalas/rotações para aproximar o domínio dos cartões."""
    h, w = img.shape[:2]
    out = []
    for _ in range(n):
        ang = rng.uniform(-6, 6)
        sc = rng.uniform(0.9, 1.1)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
        M[:, 2] += rng.uniform(-0.08, 0.08, 2) * (w, h)
        out.append(cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE))
    return out


# ---------------------------------------------------------------------------
# Modelos clássicos (OpenCV ML)
# ---------------------------------------------------------------------------
class ClassicDigitOCR:
    """Treina e testa KNN e SVM do OpenCV sobre um extrator de features (raw ou HOG)."""

    def __init__(self, feature='hog', k=3, svm_c=10.0, svm_gamma=None):
        self.feature = feature
        self.fx = FEATURES[feature]
        self.k = k
        self.svm_c = svm_c
        self.svm_gamma = svm_gamma
        self.knn = None
        self.svm = None

    def features(self, imgs):
        return np.array([self.fx(im) for im in imgs], dtype=np.float32)

    def train(self, imgs, y, augment_n=0, seed=0):
        if augment_n:
            rng = np.random.default_rng(seed)
            imgs2, y2 = list(imgs), list(y)
            for im, lab in zip(imgs, y):
                for a in augment(im, rng, augment_n):
                    imgs2.append(a)
                    y2.append(lab)
            imgs, y = imgs2, np.array(y2)
        X = self.features(imgs)
        y = np.asarray(y, dtype=np.int32)

        self.knn = cv2.ml.KNearest_create()
        self.knn.train(X, cv2.ml.ROW_SAMPLE, y)

        self.svm = cv2.ml.SVM_create()
        self.svm.setType(cv2.ml.SVM_C_SVC)
        self.svm.setKernel(cv2.ml.SVM_RBF)
        self.svm.setC(self.svm_c)
        gamma = self.svm_gamma or 1.0 / (X.shape[1] * X.var())
        self.svm.setGamma(gamma)
        self.svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 2000, 1e-6))
        self.svm.train(X, cv2.ml.ROW_SAMPLE, y)
        return self

    def predict(self, imgs, model='svm'):
        if len(imgs) == 0:
            return np.array([], dtype=int)
        X = self.features(imgs)
        if model == 'knn':
            _, res, _, _ = self.knn.findNearest(X, self.k)
        else:
            _, res = self.svm.predict(X)
        return res.ravel().astype(int)

    def evaluate(self, imgs, y):
        out = {}
        for m in ('knn', 'svm'):
            p = self.predict(imgs, m)
            out[m] = {'pred': p,
                      'accuracy': accuracy_score(y, p),
                      'f1_macro': f1_score(y, p, average='macro')}
        return out


def eval_groups(predict_fn, imgs, labels):
    """Avalia em recortes de 4 caracteres: métricas por dígito (ignorando '_') e por grupo."""
    yt, yp, grp_ok = [], [], []
    for img, lab in zip(imgs, labels):
        pred = predict_fn(img)
        n_dig = sum(c != '_' for c in lab)
        if len(pred) != len(lab) and len(pred) == n_dig:
            # OCR que não devolve as posições vazias: realinha nas posições com dígito
            it = iter(pred)
            pred = ''.join('_' if c == '_' else next(it) for c in lab)
        pred = (pred + '?' * len(lab))[:len(lab)]
        ok = True
        for c, pc in zip(lab, pred):
            if c == '_':
                continue
            yt.append(int(c))
            yp.append(int(pc) if pc.isdigit() else -1)
            ok &= (c == pc)
        grp_ok.append(ok)
    yt, yp = np.array(yt), np.array(yp)
    return {'accuracy': accuracy_score(yt, yp),
            'f1_macro': f1_score(yt, yp, average='macro', labels=list(range(10))),
            'group_accuracy': float(np.mean(grp_ok)),
            'y_true': yt, 'y_pred': yp}


# ---------------------------------------------------------------------------
# Cartão completo: utilitários de parsing e métricas
# ---------------------------------------------------------------------------
DATE_RE = re.compile(r'(?<!\d)(\d{2})\s*/\s*(\d{2}|\d{4})(?!\d)')


def normalize_date(mm, yy):
    """'2024/12' ou '12/2024' ou '12/24' -> '12/24'."""
    if len(mm) == 4:  # formato AAAA/MM
        mm, yy = yy, mm
    if not (1 <= int(mm) <= 12):
        return None
    return f'{mm}/{yy[-2:]}'


def pick_expiry(texts):
    """Escolhe a data de validade entre as datas encontradas (a mais tardia)."""
    cands = []
    for t in texts:
        for m in re.finditer(r'(?<!\d)(\d{4})\s*/\s*(\d{2})(?!\d)|(?<!\d)(\d{2})\s*/\s*(\d{2,4})(?!\d)', t):
            if m.group(1):
                d = normalize_date(m.group(1), m.group(2))
            else:
                d = normalize_date(m.group(3), m.group(4))
            if d:
                cands.append(d)
    if not cands:
        return ''
    return max(cands, key=lambda d: (int(d[3:]), int(d[:2])))


def luhn_ok(s):
    """Dígito verificador de Luhn (ISO/IEC 7812)."""
    if not s or not s.isdigit():
        return False
    t = 0
    for i, c in enumerate(reversed(s)):
        d = int(c)
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        t += d
    return t % 10 == 0


def pick_card_number(digits):
    """Escolhe o número do cartão numa sequência de dígitos lida pelo OCR: devolve a própria
    sequência se tiver 16-19 dígitos e passar no Luhn; senão a maior subsequência contígua
    (19, 16, 18, 17 dígitos) iniciada por 3/4/5/6/9 que passe no Luhn; senão a sequência original."""
    if 16 <= len(digits) <= 19 and luhn_ok(digits):
        return digits
    for n in (19, 16, 18, 17):
        for i in range(0, len(digits) - n + 1):
            sub = digits[i:i + n]
            if sub[0] in '34569' and luhn_ok(sub):
                return sub
    return digits


def lcs(a, b):
    dp = [0] * (len(b) + 1)
    for ca in a:
        prev = 0
        for j, cb in enumerate(b):
            cur = dp[j + 1]
            dp[j + 1] = prev + 1 if ca == cb else max(dp[j + 1], dp[j])
            prev = cur
    return dp[-1]


def card_metrics(df_true, df_pred):
    """Métricas de cartão completo.
    - acc_numero: número inteiro idêntico
    - f1_digitos: F1 por dígito (precisão/recall via maior subsequência comum)
    - acc_validade: validade correta (apenas cartões com validade no gabarito)"""
    m = df_true.merge(df_pred, on='arquivo', suffixes=('_true', '_pred'))
    nt = m['numero_true'].astype(str)
    npred = m['numero_pred'].fillna('').astype(str)
    common = sum(lcs(a, b) for a, b in zip(nt, npred))
    P = common / max(1, npred.str.len().sum())
    R = common / max(1, nt.str.len().sum())
    has_date = m['validade_true'].fillna('') != ''
    return {
        'acc_numero': float((nt == npred).mean()),
        'precisao_digitos': P,
        'recall_digitos': R,
        'f1_digitos': 2 * P * R / max(1e-9, P + R),
        'acc_validade': float((m.loc[has_date, 'validade_true'] == m.loc[has_date, 'validade_pred'].fillna('')).mean()),
    }


# ---------------------------------------------------------------------------
# Pipeline clássico para o cartão completo
# ---------------------------------------------------------------------------
CARD_W, CARD_H = 856, 540  # proporção ISO/IEC 7810 ID-1 (85,6 x 54 mm)


def _order_corners(pts):
    s = pts.sum(1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], np.float32)


def rectify_card(img):
    """Localiza o cartão (Canny + maior contorno retangular) e corrige a perspectiva
    para 856x540. Se não encontrar, usa a imagem inteira."""
    s = 800 / max(img.shape[:2])
    g = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(img, None, fx=s, fy=s), cv2.COLOR_BGR2GRAY), (5, 5), 0)
    quad = None
    for lo, hi in [(30, 90), (10, 50), (50, 150)]:
        e = cv2.dilate(cv2.Canny(g, lo, hi), np.ones((5, 5), np.uint8))
        cs, _ = cv2.findContours(e, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(cs, key=cv2.contourArea, reverse=True)[:5]:
            a = cv2.contourArea(c)
            if a < 0.2 * g.size:
                break
            r = cv2.minAreaRect(c)
            w, h = r[1]
            ar = max(w, h) / max(1, min(w, h))
            if 1.3 < ar < 1.9 and a / (w * h) > 0.8:
                hull = cv2.convexHull(c)
                ap = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
                quad = (ap.reshape(-1, 2) if len(ap) == 4 else cv2.boxPoints(r)).astype(np.float32) / s
                break
        if quad is not None:
            break
    if quad is None:
        h, w = img.shape[:2]
        quad = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    q = _order_corners(quad)
    if np.linalg.norm(q[3] - q[0]) > np.linalg.norm(q[1] - q[0]):  # cartão na vertical
        q = np.roll(q, -1, axis=0)
    M = cv2.getPerspectiveTransform(q, np.float32([[0, 0], [CARD_W, 0], [CARD_W, CARD_H], [0, CARD_H]]))
    return cv2.warpPerspective(img, M, (CARD_W, CARD_H)), q


def _gradx(card):
    g = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    return np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))


def find_text_rows(gx, dh, ymin, ymax):
    """Candidatos a linha de texto: picos de energia de gradiente horizontal combinada
    com a cobertura (fração das colunas com traços). Retorna centros ordenados de cima p/ baixo."""
    x0, x1 = int(0.04 * CARD_W), int(0.96 * CARD_W)
    win = np.convolve(gx[:, x0:x1].mean(1), np.ones(dh) / dh, 'same')
    colE = cv2.blur(gx, (15, dh))[:, x0:x1]
    cov = (colE > np.percentile(colE, 70)).mean(1)
    ys = np.arange(CARD_H)
    band = (ys > ymin * CARD_H) & (ys < ymax * CARD_H)
    if not band.any():
        return [], None
    sc = 0.5 * win / win[band].max() + 0.5 * cov / max(cov[band].max(), 1e-6)
    best = sc[band].max()
    cand = [int(y) for y in ys[band]
            if sc[y] >= 0.85 * best and sc[y] == sc[max(0, y - dh // 2):y + dh // 2].max()]
    return cand, sc


def segment_row(gx, y0, y1, xmin=0.03, xmax=0.97, n_target=(16, 19)):
    """Segmenta uma faixa em caracteres pela projeção vertical do gradiente.
    O passo (largura de um caractere) é escolhido para que o total feche em 16 ou 19 dígitos."""
    h = y1 - y0
    prof = np.convolve(gx[y0:y1].mean(0), np.ones(5) / 5, 'same')
    x0, x1 = int(xmin * CARD_W), int(xmax * CARD_W)
    base, peak = np.percentile(prof[x0:x1], 20), np.percentile(prof[x0:x1], 95)
    on = prof > base + 0.25 * (peak - base)
    on[:x0] = False
    on[x1:] = False
    segs, i = [], 0
    while i < CARD_W:
        if on[i]:
            j = i
            while j < CARD_W and on[j]:
                j += 1
            segs.append([i, j])
            i = j
        else:
            i += 1

    def groups_for(pitch):
        grp = []
        for s in segs:
            if grp and s[0] - grp[-1][1] < 0.45 * pitch:
                grp[-1][1] = s[1]
            else:
                grp.append(list(s))
        return [g for g in grp if g[1] - g[0] > 0.6 * pitch]

    nominal = 0.65 * h
    pitch = nominal
    if n_target:
        best = None
        for p in np.linspace(0.5 * h, 0.9 * h, 41):
            n = sum(max(1, round((b - a) / p)) for a, b in groups_for(p))
            if n in n_target and (best is None or abs(p - nominal) < abs(best - nominal)):
                best = p
        pitch = best or nominal
    boxes = []
    for a, b in groups_for(pitch):
        n = max(1, int(round((b - a) / pitch)))
        st = (b - a) / n
        boxes.append([(int(a + k * st), int(a + (k + 1) * st)) for k in range(n)])
    return boxes  # lista de grupos; cada grupo = lista de (x0, x1)


def _slice(card, y0, y1, xa, xb, pad=3):
    return card[max(0, y0):y1, max(0, xa - pad):min(CARD_W, xb + pad)]


class ClassicCardOCR:
    """Pipeline clássico: retificação do cartão -> localização da linha do número ->
    segmentação por projeção -> classificação HOG + KNN/SVM (OpenCV)."""

    DIGIT_H = 46   # altura da faixa do número no cartão retificado (igual aos recortes do ds2)
    DATE_H = 32    # altura típica da validade

    def __init__(self, digit_ocr, model='svm'):
        self.ocr = digit_ocr
        self.model = model

    def _classify(self, imgs):
        return ''.join(map(str, self.ocr.predict(imgs, self.model)))

    def read_number(self, card, gx):
        cand, _ = find_text_rows(gx, self.DIGIT_H, 0.42, 0.80)
        if not cand:
            return '', None, []
        yc = min(cand)
        y0, y1 = yc - self.DIGIT_H // 2, yc + self.DIGIT_H // 2
        groups = segment_row(gx, y0, y1)
        flat = [b for g in groups for b in g]
        digits = self._classify([_slice(card, y0, y1, a, b) for a, b in flat])
        return digits, (y0, y1), flat

    def read_date(self, card, gx, below):
        cands, _ = find_text_rows(gx, self.DATE_H, below / CARD_H + 0.01, 0.95)
        dates = []
        for yc in cands[:3]:
            y0, y1 = yc - self.DATE_H // 2, yc + self.DATE_H // 2
            for g in segment_row(gx, y0, y1, n_target=None):
                if len(g) == 5:  # MM/AA
                    d = self._classify([_slice(card, y0, y1, a, b, 2) for k, (a, b) in enumerate(g) if k != 2])
                    nd = normalize_date(d[:2], d[2:])
                    if nd:
                        dates.append(nd)
        return pick_expiry(dates)

    def read(self, img):
        card, quad = rectify_card(img)
        gx = _gradx(card)
        number, rng, boxes = self.read_number(card, gx)
        date = self.read_date(card, gx, rng[1]) if rng else ''
        return {'numero': number, 'validade': date, 'card': card, 'quad': quad,
                'line': rng, 'boxes': boxes}


# ---------------------------------------------------------------------------
# Modelo pré-treinado: EasyOCR (PyTorch; detector CRAFT + reconhecedor CRNN)
# ---------------------------------------------------------------------------
def _group_lines(results):
    """Agrupa as caixas do EasyOCR em linhas (mesma altura) e ordena da esquerda p/ direita."""
    items = []
    for box, txt, conf in results:
        b = np.array(box)
        (x0, y0), (x1, y1) = b.min(0), b.max(0)
        items.append({'x0': x0, 'yc': (y0 + y1) / 2, 'h': y1 - y0, 't': txt})
    items.sort(key=lambda d: d['yc'])
    lines = []
    for it in items:
        for L in lines:
            if abs(L[-1]['yc'] - it['yc']) < 0.5 * max(L[-1]['h'], it['h']):
                L.append(it)
                break
        else:
            lines.append([it])
    return [sorted(L, key=lambda d: d['x0']) for L in lines]


class PretrainedCardOCR:
    """OCR com o EasyOCR pré-treinado.
    mode='full'  : detecção + reconhecimento na foto inteira;
    mode='hybrid': retificação/localização clássica da linha do número + reconhecimento EasyOCR."""

    def __init__(self, gpu=False, mode='hybrid'):
        import easyocr
        self.reader = easyocr.Reader(['en'], gpu=gpu, verbose=False)
        self.mode = mode

    def recognize_digits(self, img):
        """Reconhece só dígitos num recorte já localizado (dígito isolado, grupo ou faixa)."""
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        h, w = g.shape
        res = self.reader.recognize(g, horizontal_list=[[0, w, 0, h]], free_list=[],
                                    allowlist='0123456789', detail=0)
        return re.sub(r'\D', '', ''.join(res))

    def _number_from_lines(self, lines):
        best, best_score = '', -1
        for L in lines:
            s = re.sub(r'\D', '', ''.join(d['t'] for d in L if '/' not in d['t']))
            if len(s) < 12:
                continue
            s = pick_card_number(s)
            score = np.median([d['h'] for d in L]) * (1 + luhn_ok(s))
            if score > best_score:
                best, best_score = s, score
        return best

    def read(self, img):
        card, quad = rectify_card(img)
        if self.mode == 'full':
            s = 1200 / max(img.shape[:2])
            res = self.reader.readtext(cv2.resize(img, None, fx=s, fy=s), allowlist='0123456789/')
            number = self._number_from_lines(_group_lines(res))
            line = None
        else:
            gx = _gradx(card)
            cand, _ = find_text_rows(gx, ClassicCardOCR.DIGIT_H, 0.42, 0.80)
            yc = min(cand) if cand else int(0.6 * CARD_H)
            line = (max(0, yc - 30), yc + 30)
            strip = card[line[0]:line[1]]
            number = pick_card_number(self.recognize_digits(strip))
            # recorta a faixa na extensão dos grupos segmentados (remove ruído das bordas)
            groups = segment_row(gx, yc - 23, yc + 23)
            if groups:
                xa = max(0, groups[0][0][0] - 12)
                xb = min(CARD_W, groups[-1][-1][1] + 12)
                trimmed = pick_card_number(self.recognize_digits(strip[:, xa:xb]))
                if (16 <= len(trimmed) <= 19 and luhn_ok(trimmed)) or not (16 <= len(number) <= 19 and luhn_ok(number)):
                    number = trimmed
            # validade: detecção + reconhecimento só na região abaixo do número
            roi = cv2.resize(card[yc + 15:], None, fx=1.5, fy=1.5)
            res = self.reader.readtext(roi, allowlist='0123456789/')
        date = pick_expiry([r[1] for r in res])
        return {'numero': number, 'validade': date, 'card': card, 'quad': quad, 'line': line}
