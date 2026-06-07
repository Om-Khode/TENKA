"""
browser_dom.py — Accessibility-tree perception for the DOM-aware planner.

Reads a page's interactive elements via a single Playwright `evaluate()`
round-trip and returns a list of `ElementInfo` records the DOM planner can
prompt against. Each element gets a stable, content-addressed `ref` ID that
the executor uses to map back to a Playwright `Locator`.

Why one big evaluate() instead of N locator queries:
  - One IPC round-trip vs N: ~10x faster on a 20-element form.
  - The JS query has full DOM access in one pass — no synchronization needed.
  - Reduces token cost per task: the perceiver is the *only* DOM probe per
    batch; everything downstream operates on the snapshot.

Ref scheme (load-bearing):
  ref = sha1(role|name|placeholder|bounds_quantized).hexdigest()[:10]
  Quantizing bounds to 8-px buckets means a 1-px reflow doesn't change the ref.
  Collisions get a `:N` suffix in DOM order — rare in practice (would require
  two elements with identical role+name+placeholder+8px-bucketed bounds).

Locator strategy:
  JS marks each captured element with `data-tenka-idx` (0,1,2,...). Python
  builds a Playwright Locator via `[data-tenka-idx="N"]`. Indices are
  per-perception (not stable across reads); refs ARE stable across reads
  when content is unchanged.

Cache (per-page, TTL'd):
  Invalidated by:
    1. `invalidate_tree_cache(page)` — explicit
    2. TTL expiry (`BROWSER_DOM_CACHE_TTL`)
    3. Caller's responsibility on click/press/navigation (we don't observe
       page mutations from this module).

Token budget:
  The serialized tree must fit in `BROWSER_DOM_TREE_TOKEN_BUDGET` (default
  4000) in the planner prompt. Truncation order, applied in sequence until
  the budget is met:
    1. Drop `bounds` from every element (~15 tok/elem).
    2. Drop `placeholder` for elements where `name` is non-empty.
    3. Prune off-viewport elements.
    4. Prune from the tail and add a `_truncated: N` marker.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Literal, Optional

from ... import config

logger = logging.getLogger("browser_dom")


# ─── Public dataclasses ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class ElementInfo:
    """
    One interactive element on the page. Frozen so callers can pass it
    around without worrying about mutation; the cache also holds these
    safely as values.

    Field meanings:
      ref          — stable content-addressed ID (see module docstring)
      role         — ARIA role (textbox, button, combobox, etc.) — explicit
                     `role=` attr wins; otherwise inferred from tag+type
      name         — accessible name (label, aria-label, aria-labelledby
                     resolution, placeholder fallback, textContent fallback)
      placeholder  — `placeholder` attribute, '' when absent
      value        — current value (text inputs / select selected option /
                     checkbox checked-as-string)
      options      — only populated for native `<select>` and (when
                     `open_comboboxes=True`) opened ARIA listboxes
      bounds       — (x, y, w, h) in CSS pixels, rounded to int
      visible      — geometry > 0 AND display!=none AND visibility!=hidden
                     AND opacity > 0
      enabled      — not [disabled] / not aria-disabled='true'
      type         — input type attr ('text', 'email', 'tel', etc.)
      tag          — HTML tag, lowercased
      form_id      — stable id for the element's `<form>` ancestor, of
                     shape "form-N" (N = index in document.forms). Empty
                     when the element has no form ancestor. Used by the
                     orchestrator to group elements per-form for multi-form
                     pages (header CTA + footer + modal).
      in_dialog    — true when the element has a `<dialog>` /
                     role="dialog" / role="alertdialog" ancestor. The
                     orchestrator uses this to prefer modal forms when
                     multiple forms exist on a page (modal = active focus).
      aria_invalid — true ONLY when the element has an explicit
                     `aria-invalid="true"` attribute. This is the signal that
                     custom-validating forms (Webflow, React) set after a
                     failed submit; it's reliable as a "form just rejected
                     this field" indicator. The prior contract also matched
                     the HTML5 `:invalid` pseudo-class as a fallback, but
                     `:invalid` fires on any unmet HTML5 constraint
                     (`required`, `pattern`, `minlength`, `type=email`)
                     even when the form's visible UI accepts the value. On
                     Webflow forms with strict patterns this generated 6+
                     spurious synthetic errors per field — drowning the one
                     real rejection. The pseudo-class is no longer consulted;
                     pure-HTML5 forms relying solely on built-in tooltips
                     fall outside this contract.
    """
    ref: str
    role: str
    name: str
    placeholder: str
    value: str
    options: tuple[str, ...]
    bounds: tuple[int, int, int, int]
    visible: bool
    enabled: bool
    type: str
    tag: str
    form_id: str = ""
    in_dialog: bool = False
    aria_invalid: bool = False
    autocomplete: str = ""


@dataclass(frozen=True)
class ValidationError:
    """
    One validation-error signal scraped from the page.

    `field_ref` — ref of the input the error anchors to. Empty when no
                  anchor could be inferred (page-level alert).
    `message`   — user-visible error text (whitespace-collapsed, truncated
                  to 300 chars).
    `source`    — provenance tag for diagnostics:
                    "aria-invalid"   — input flagged aria-invalid=true,
                                       no paired alert text found
                    "text-match"     — fallback: anchored by token overlap
                                       between error text and a captured
                                       field's name+type+placeholder (with
                                       synonym expansion). Used when DOM
                                       proximity hit an ambiguous form-level
                                       ancestor.
                    "html5-invalid"  — DEPRECATED: input matches
                    "alert"          — role="alert" / aria-live=assertive
                    "describedby"    — element referenced via aria-describedby
                                       on an aria-invalid input
                    "error-class"    — sibling/descendant with class*="error"
                                       or class*="invalid"
    """
    field_ref: str
    message: str
    source: str


@dataclass
class PageDomTree:
    """
    Result of a successful `read_page_dom` call.

    `elements`         — captured ElementInfo list
    `ref_to_locator`   — map for the executor to convert ref → Playwright Locator
    `truncated`        — count of elements pruned by the token-budget pass; 0
                         means the tree fit within budget
    `read_at`          — time.monotonic() when this tree was perceived; used by
                         the cache TTL logic
    `viewport`         — (width, height) of the page viewport at read time
    `validation_errors` — validation signals collected during the same JS
                          pass. Anchored to a captured input's ref when
                          possible. The orchestrator forwards these as
                          planner feedback after a submit click.
    `evaluate_failed`   — Post-nav-detection: True when `page.evaluate` raised
                          mid-perception (Playwright signals a navigation in
                          flight via "Execution context was destroyed"). The
                          orchestrator uses this as a strong success signal
                          when it fires AFTER a submit click landed: the page
                          is navigating to a thank-you URL.
    `url`               — Page URL captured at perception time (best-effort,
                          empty when unavailable). The orchestrator compares
                          across perceptions to detect hard navigations that
                          completed BEFORE the next perception ran.
    """
    elements: list[ElementInfo]
    ref_to_locator: dict[str, Any]  # str ref → Playwright Locator
    truncated: int = 0
    read_at: float = 0.0
    viewport: tuple[int, int] = (0, 0)
    validation_errors: tuple[ValidationError, ...] = ()
    evaluate_failed: bool = False
    url: str = ""


# ─── Module-level cache ───────────────────────────────────────────────────────

# Keyed by id(page) — Page objects don't have a hashable identity Playwright
# guarantees, so we use Python's object id. Entries clear when invalidated
# or evicted by TTL.
_tree_cache: dict[int, PageDomTree] = {}


def invalidate_tree_cache(page: Any) -> None:
    """
    Drop the cached tree for a specific page. Called by the executor after
    click/press/navigation actions that may mutate the DOM.

    Safe to call when no cache entry exists.
    """
    _tree_cache.pop(id(page), None)


def reset_state_for_test() -> None:
    """Test helper. NEVER call from production paths."""
    _tree_cache.clear()


# ─── Filter modes ─────────────────────────────────────────────────────────────

FilterMode = Literal["all", "interactive", "form"]


# ─── The DOM-side query (one big evaluate) ────────────────────────────────────

# Single JS function. Returns a list of dict objects, one per captured element.
# Each captured element also gets `data-tenka-idx="N"` set so Python can build
# a Playwright Locator without needing CSS-path generation.
#
# The function takes one parameter `cfg` which carries the filter mode and
# whether to enumerate combobox options eagerly.
_DOM_QUERY_JS = r"""
(cfg) => {
    const FILTER = cfg.filter || "interactive";
    const OPEN_COMBOBOXES = !!cfg.openComboboxes;

    const INTERACTIVE_TAGS = new Set([
        'input', 'select', 'textarea', 'button',
    ]);
    const INTERACTIVE_ROLES = new Set([
        'textbox', 'button', 'link', 'checkbox', 'radio', 'combobox',
        'listbox', 'menuitem', 'switch', 'tab', 'slider', 'searchbox',
        'spinbutton', 'option',
    ]);
    // Roles/tags we accept in "all" but not "interactive"
    const PRESENTATIONAL_ROLES = new Set([
        'presentation', 'none', 'group', 'region', 'main', 'banner',
        'navigation', 'complementary', 'contentinfo', 'heading', 'paragraph',
        'list', 'listitem', 'separator', 'status', 'log', 'timer',
    ]);

    function impliedRole(el) {
        const tag = el.tagName.toLowerCase();
        const explicit = el.getAttribute('role');
        if (explicit) return explicit;
        if (tag === 'a') return el.hasAttribute('href') ? 'link' : '';
        if (tag === 'button') return 'button';
        if (tag === 'select') return 'combobox';
        if (tag === 'textarea') return 'textbox';
        if (tag === 'input') {
            const t = (el.type || 'text').toLowerCase();
            if (t === 'checkbox') return 'checkbox';
            if (t === 'radio') return 'radio';
            if (t === 'submit' || t === 'button' || t === 'reset' || t === 'image') return 'button';
            if (t === 'range') return 'slider';
            if (t === 'search') return 'searchbox';
            if (t === 'number') return 'spinbutton';
            if (t === 'hidden') return '';  // intentionally excluded
            return 'textbox';
        }
        if (el.isContentEditable) return 'textbox';
        return '';
    }

    function isInteractiveByMode(role, tag, el) {
        if (FILTER === 'all') {
            return !!role && !PRESENTATIONAL_ROLES.has(role);
        }
        // "interactive" and "form" share the same gate;
        // "form" prunes further below.
        if (INTERACTIVE_TAGS.has(tag)) return true;
        if (INTERACTIVE_ROLES.has(role)) return true;
        if (el.isContentEditable) return true;
        if (el.hasAttribute('tabindex')) {
            const ti = el.getAttribute('tabindex');
            if (ti && ti !== '-1') return true;
        }
        return false;
    }

    function _labelTextStripped(label) {
        // Clone-and-strip pattern: a wrapping <label>Country <select>...</select></label>
        // returns the entire label text INCLUDING all <option> text from
        // its select child via .textContent. We need just "Country".
        // Cloning is cheap (label is small); strip form controls; then read.
        try {
            const clone = label.cloneNode(true);
            const sel = 'input, select, textarea, button, [role="textbox"], '
                      + '[role="combobox"], [role="button"], [role="checkbox"], '
                      + '[role="radio"], [role="listbox"]';
            clone.querySelectorAll(sel).forEach(n => n.remove());
            return (clone.textContent || '').trim().replace(/\s+/g, ' ');
        } catch (e) {
            return '';
        }
    }

    function accessibleName(el) {
        // Simplified ARIA naming algorithm. Order matters.
        const aria = (el.getAttribute('aria-label') || '').trim();
        if (aria) return aria;
        const labelledby = el.getAttribute('aria-labelledby');
        if (labelledby) {
            const ids = labelledby.split(/\s+/).filter(Boolean);
            const parts = [];
            for (const id of ids) {
                const ref = document.getElementById(id);
                if (ref) parts.push((ref.textContent || '').trim());
            }
            const joined = parts.join(' ').trim();
            if (joined) return joined;
        }
        // Form-control native labels (input/select/textarea/button only)
        if (el.labels && el.labels.length > 0) {
            const t = _labelTextStripped(el.labels[0]);
            if (t) return t;
        }
        // For ANY element with id, also check <label for="id"> — handles
        // custom widgets (role="combobox" on <div>) that aren't in
        // el.labels (browsers only populate that for form-associated tags).
        if (el.id) {
            try {
                const labelFor = document.querySelector('label[for="' + el.id.replace(/"/g, '\\"') + '"]');
                if (labelFor) {
                    const t = _labelTextStripped(labelFor);
                    if (t) return t;
                }
            } catch (e) { /* invalid selector — skip */ }
        }
        // Title attr
        const title = (el.getAttribute('title') || '').trim();
        if (title) return title;
        // Visible textContent for naming-from-content roles
        const tag = el.tagName.toLowerCase();
        const explicit = el.getAttribute('role') || '';
        if (tag === 'button' || tag === 'a'
            || explicit === 'button' || explicit === 'link'
            || explicit === 'menuitem' || explicit === 'tab'
            || explicit === 'option' || explicit === 'treeitem') {
            const t = (el.textContent || '').trim().replace(/\s+/g, ' ');
            if (t) return t;
        }
        // alt for image-style inputs
        const alt = (el.getAttribute('alt') || '').trim();
        if (alt) return alt;
        return '';
    }

    function comboboxFallbackName(el) {
        // For combobox <input> elements with no accessible name, extract
        // a name from the widget container.  Catches placeholder text from
        // react-select ("Select State"), MUI Autocomplete, Headless UI, etc.
        // Strategy 1: sibling text in the value/control container
        let cur = el.parentElement;
        for (let i = 0; i < 4 && cur; i++) {
            for (const child of cur.children) {
                if (child === el || child.contains(el)) continue;
                if (child.querySelector('input,select,textarea,button,[role="button"]')) continue;
                if (child.tagName === 'svg' || child.tagName === 'SVG') continue;
                const t = (child.textContent || '').trim().replace(/\s+/g, ' ');
                if (t && t.length > 1 && t.length < 80) return t;
            }
            cur = cur.parentElement;
        }
        // Strategy 2: nearest ancestor with a descriptive id
        cur = el.parentElement;
        for (let i = 0; i < 8 && cur && cur !== document.body; i++) {
            if (cur.id) {
                const clean = cur.id
                    .replace(/([a-z])([A-Z])/g, '$1 $2')
                    .replace(/[-_]/g, ' ')
                    .replace(/\b(container|wrapper|field|group|div|col|row|section)\b/gi, '')
                    .trim();
                if (clean) return clean;
            }
            cur = cur.parentElement;
        }
        return '';
    }

    function isVisible(el, rect) {
        if (rect.width <= 0 || rect.height <= 0) return false;
        const cs = window.getComputedStyle(el);
        if (cs.display === 'none' || cs.visibility === 'hidden') return false;
        const op = parseFloat(cs.opacity || '1');
        if (!isNaN(op) && op <= 0) return false;
        return true;
    }

    function nearestForm(el) {
        let cur = el;
        while (cur && cur !== document.body) {
            if (cur.tagName && cur.tagName.toLowerCase() === 'form') return cur;
            cur = cur.parentElement;
        }
        return null;
    }

    // Optional: expand role=combobox listboxes by clicking, snapshotting,
    // and clicking again to close. Disabled by default — too disruptive
    // to do unconditionally. Caller opts in via cfg.openComboboxes.
    // We don't actually click here — that side-effect belongs in the
    // executor, not the perceiver. Custom-combobox option enumeration
    // is left to the orchestrator's open-then-reperceive flow.

    // FORM filter: find the nearest form ancestor of the focused element
    // (or first form on page) and only emit its descendants.
    let formRoot = null;
    if (FILTER === 'form') {
        const focused = document.activeElement;
        formRoot = (focused && focused !== document.body) ? nearestForm(focused) : null;
        if (!formRoot) {
            const allForms = document.getElementsByTagName('form');
            if (allForms.length > 0) formRoot = allForms[0];
        }
    }

    // Form-id assignment for multi-form disambiguation. document.forms is
    // a live HTMLCollection — we materialize once, then map each captured
    // element to its enclosing form's index. Stable across re-perceptions
    // as long as form ordering doesn't change.
    const formsList = Array.from(document.forms || []);
    function formIdFor(el) {
        const f = nearestForm(el);
        if (!f) return '';
        const idx = formsList.indexOf(f);
        return idx >= 0 ? ('form-' + idx) : '';
    }

    // Dialog/modal ancestor detection. Walk up looking for <dialog>,
    // role="dialog", or role="alertdialog". Used by the orchestrator to
    // prefer modal forms over background page forms.
    function inDialog(el) {
        let cur = el;
        while (cur && cur !== document.body) {
            if (cur.tagName) {
                const tag = cur.tagName.toLowerCase();
                if (tag === 'dialog') return true;
            }
            const role = cur.getAttribute && cur.getAttribute('role');
            if (role === 'dialog' || role === 'alertdialog') return true;
            cur = cur.parentElement;
        }
        return false;
    }

    // Walk all elements once. We use querySelectorAll('*') because
    // attribute-shaped queries miss implicit-role inputs when role-attr
    // isn't set, and tag-shaped queries miss [role="..."] divs.
    const allEls = document.querySelectorAll('*');
    const out = [];
    let idx = 0;

    for (const el of allEls) {
        if (formRoot && !formRoot.contains(el)) continue;
        const tag = el.tagName.toLowerCase();
        const role = impliedRole(el);
        if (!role) continue;
        if (!isInteractiveByMode(role, tag, el)) continue;
        // Hidden inputs handled inside impliedRole returning ''
        if (tag === 'input' && (el.type || '').toLowerCase() === 'hidden') continue;

        const rect = el.getBoundingClientRect();
        const visible = isVisible(el, rect);
        // We INCLUDE invisible elements in the tree but flag them so the
        // planner can decide. Off-viewport + invisible are different signals.
        // (Token-budget pass prunes them later if budget tight.)

        // Get options for native <select>
        let options = [];
        if (tag === 'select') {
            try {
                options = Array.from(el.options || []).map(o => (o.text || o.value || '').trim());
                options = options.filter(t => t.length > 0).slice(0, 50);
            } catch (e) {
                options = [];
            }
        }

        // value: input.value, select.value (already serialized), checkbox state
        let value = '';
        if ('value' in el) {
            value = String(el.value || '');
        }
        if (tag === 'input') {
            const t = (el.type || '').toLowerCase();
            if (t === 'checkbox' || t === 'radio') {
                value = el.checked ? 'on' : 'off';
            }
        }

        // Mark with sequential idx so Python can build a Locator
        try {
            el.dataset.tenkaIdx = String(idx);
        } catch (e) {
            // Some elements (e.g. SVG in older browsers) reject dataset.
            // Skip them — we can't reliably locate them anyway.
            continue;
        }

        // aria-invalid signal. ONLY matches the explicit `aria-invalid="true"`
        // attribute — the signal Webflow/React forms set when their JS
        // validation layer rejects a field after submit. The HTML5 `:invalid`
        // pseudo-class was deliberately removed from this contract: it fires
        // for any unmet `required`/`pattern`/`minlength`/`type=email`
        // constraint, even on fields the form's user-visible UI accepts. On
        // strict-pattern Webflow forms this generated 6+ spurious synthetic
        // errors via the loop below, drowning the one real rejection in noise.
        const ariaInvalidAttr = (el.getAttribute('aria-invalid') || '').toLowerCase();
        const ariaInvalid = (ariaInvalidAttr === 'true');

        let elName = accessibleName(el).slice(0, 200);
        if (!elName && role === 'combobox') {
            elName = comboboxFallbackName(el).slice(0, 200);
        }

        out.push({
            idx: idx,
            tag: tag,
            role: role,
            name: elName,
            placeholder: (el.getAttribute('placeholder') || '').slice(0, 200),
            value: value.slice(0, 500),
            options: options,
            bounds: [
                Math.round(rect.left),
                Math.round(rect.top),
                Math.round(rect.width),
                Math.round(rect.height),
            ],
            visible: visible,
            enabled: !el.disabled && el.getAttribute('aria-disabled') !== 'true',
            type: (el.type || '').toLowerCase(),
            form_id: formIdFor(el),
            in_dialog: inDialog(el),
            aria_invalid: ariaInvalid,
            // describedby → list of element ids whose text we should treat as
            // this field's error message (when aria_invalid). Captured here
            // so Python can correlate without a second DOM walk.
            describedby: (el.getAttribute('aria-describedby') || '')
                .split(/\s+/).filter(Boolean).slice(0, 8),
            // Element id (when present) — used to identify the captured field
            // when an alert references it via aria-controls / for=.
            el_id: el.id || '',
            autocomplete: (role === 'combobox')
                ? (el.getAttribute('aria-autocomplete') || '').toLowerCase()
                : '',
        });
        idx++;
    }

    // ─── Collect validation-error elements ─────────────────────────────────
    // Selector covers the four documented sources. We deliberately do NOT
    // match elements with NO visible text — empty error containers are
    // skeleton placeholders that React renders but only fills on validation
    // failure; their *presence* is not the signal, *content* is.
    const ERR_SELECTOR = (
        '[role="alert"], [aria-live="assertive"], '
        + '[class*="error"]:not([class*="error-icon"]):not([class*="errorless"]), '
        + '[class*="invalid"]'
    );
    function _txt(node) {
        return ((node.textContent || '').trim().replace(/\s+/g, ' '));
    }
    function _isVis(node) {
        const r = node.getBoundingClientRect();
        if (r.width <= 0 || r.height <= 0) return false;
        const cs = window.getComputedStyle(node);
        if (cs.display === 'none' || cs.visibility === 'hidden') return false;
        const op = parseFloat(cs.opacity || '1');
        if (!isNaN(op) && op <= 0) return false;
        return true;
    }
    // Build idx → captured-record lookup so we can map error→idx via
    // aria-describedby ref_to_id resolution later in Python.
    const idToIdx = {};
    for (const cap of out) {
        if (cap.el_id) idToIdx[cap.el_id] = cap.idx;
    }
    const validationErrors = [];
    const seenErrEls = new WeakSet();
    let errEls = [];
    try {
        errEls = Array.from(document.querySelectorAll(ERR_SELECTOR));
    } catch (e) { errEls = []; }
    for (const errEl of errEls) {
        if (seenErrEls.has(errEl)) continue;
        seenErrEls.add(errEl);
        if (!_isVis(errEl)) continue;
        const txt = _txt(errEl);
        // Skip empties, very short flickers ("!"), and absurdly long blobs
        // (a header that happens to use class="error-banner" hosting
        // navigation isn't a validation message).
        if (txt.length < 2 || txt.length > 300) continue;

        // Resolve anchor to a captured field idx.
        let fieldIdx = -1;
        let source = 'error-class';
        const role = (errEl.getAttribute('role') || '').toLowerCase();
        const ariaLive = (errEl.getAttribute('aria-live') || '').toLowerCase();
        if (role === 'alert' || ariaLive === 'assertive') source = 'alert';

        // 1. Some captured field references this errEl via aria-describedby.
        if (errEl.id && Object.prototype.hasOwnProperty.call(idToIdx, errEl.id) === false) {
            // (errEl is the describer, not the described — search the inverse.)
        }
        if (errEl.id) {
            for (const cap of out) {
                if (cap.describedby && cap.describedby.indexOf(errEl.id) >= 0) {
                    fieldIdx = cap.idx;
                    if (cap.aria_invalid) source = 'describedby';
                    break;
                }
            }
        }
        // 2. errEl carries `for=` or `aria-controls` pointing to a captured field.
        if (fieldIdx < 0) {
            const forAttr = errEl.getAttribute('for') || errEl.getAttribute('aria-controls');
            if (forAttr) {
                const ids = forAttr.split(/\s+/).filter(Boolean);
                for (const i of ids) {
                    if (Object.prototype.hasOwnProperty.call(idToIdx, i)) {
                        fieldIdx = idToIdx[i];
                        break;
                    }
                }
            }
        }
        // 3. DOM proximity: walk up parents looking for a captured input.
        //    The walker MUST stop at the first ancestor that contains
        //    MULTIPLE captures — that means the error sits at form/section
        //    level (e.g. an alert at the bottom of the form, below the
        //    submit button), and `querySelector('[data-tenka-idx]')` would
        //    arbitrarily pick the FIRST descendant in DOM order (which is
        //    the topmost form field, not the field that failed).
        //    Single-capture ancestors are still trustworthy — that's the
        //    "error sibling next to its input" case.
        if (fieldIdx < 0) {
            let cur = errEl.parentElement;
            let hops = 0;
            while (cur && cur !== document.body && hops < 6) {
                const captures = (cur.querySelectorAll
                    ? cur.querySelectorAll('[data-tenka-idx]') : []);
                if (captures.length === 1) {
                    const i = parseInt(captures[0].dataset.tenkaIdx || '-1', 10);
                    if (!isNaN(i) && i >= 0) {
                        fieldIdx = i;
                        break;
                    }
                } else if (captures.length > 1) {
                    // Ambiguous — error is form-level, not field-level.
                    // Stop ascending; let the text-match fallback decide.
                    break;
                }
                cur = cur.parentElement;
                hops++;
            }
        }

        // 3b. Text-match fallback. When DOM proximity gave no answer (or
        //     stopped at an ambiguous ancestor), score each captured field
        //     by token overlap between the error message and the field's
        //     name + type + placeholder. Best score wins. Generic across
        //     forms — "Please enter a valid phone number." matches a field
        //     whose name contains "Number" or whose type is "tel". Falls
        //     through to page-level when no field scores.
        if (fieldIdx < 0) {
            const errToks = new Set();
            const errLower = txt.toLowerCase();
            const errSplit = errLower.split(/[^a-z0-9]+/);
            for (const t of errSplit) {
                if (t.length >= 4) errToks.add(t);
            }
            // Tiny generic alias map — same shape as Python's _FIELD_ALIASES,
            // duplicated here because the JS pass runs in the page context
            // and can't import from Python. Keep in sync with browser_dom_
            // orchestrator.py's _FIELD_ALIASES; entries are bidirectional.
            const aliasGroups = [
                ['mobile', 'phone', 'contact', 'tel', 'telephone', 'cell', 'cellphone', 'number'],
                ['email', 'mail'],
                ['name', 'first', 'last', 'full'],
                ['company', 'organization', 'business', 'employer'],
                ['address', 'street', 'location'],
                ['zip', 'postal', 'postcode'],
                ['country', 'nation', 'region'],
                ['password', 'passcode'],
            ];
            // Expand error tokens by alias map.
            const errExpanded = new Set(errToks);
            for (const grp of aliasGroups) {
                for (const t of grp) {
                    if (errToks.has(t)) {
                        for (const u of grp) errExpanded.add(u);
                        break;
                    }
                }
            }
            let bestScore = 0;
            let bestIdx = -1;
            for (const cap of out) {
                const haystack = ((cap.name || '') + ' '
                    + (cap.type || '') + ' '
                    + (cap.placeholder || '')).toLowerCase();
                const fieldToks = haystack.split(/[^a-z0-9]+/).filter(t => t.length >= 3);
                let score = 0;
                for (const ft of fieldToks) {
                    if (errExpanded.has(ft)) score++;
                }
                if (score > bestScore) {
                    bestScore = score;
                    bestIdx = cap.idx;
                }
            }
            if (bestScore > 0) {
                fieldIdx = bestIdx;
                source = 'text-match';
            }
        }

        validationErrors.push({
            field_idx: fieldIdx,
            message: txt.slice(0, 300),
            source: source,
        });
    }

    // 4. Synthetic entries for aria-invalid fields with no paired message.
    //    These let the planner know the field needs another fix even when
    //    the site swallowed the visible alert text.
    const claimedIdxs = new Set(
        validationErrors.filter(e => e.field_idx >= 0).map(e => e.field_idx)
    );
    for (const cap of out) {
        if (!cap.aria_invalid) continue;
        if (claimedIdxs.has(cap.idx)) continue;
        validationErrors.push({
            field_idx: cap.idx,
            message: '(field flagged invalid; no error text exposed)',
            source: 'aria-invalid',
        });
    }

    // Dedupe errors with identical (field_idx, message).
    // The ERR_SELECTOR matches `[class*="error"]` which can hit BOTH an
    // outer wrapper (e.g. class="error-msg-wrapper") AND a nested inner
    // element (class="error-text") — both contain the same visible text,
    // both anchor to the same field, but seenErrEls only dedupes by DOM
    // identity so two distinct elements both produce error entries. The
    // signature-based pass below collapses these to one entry.
    const dedupedErrors = [];
    const seenSigs = new Set();
    for (const ve of validationErrors) {
        const sig = ve.field_idx + '::' +
            (ve.message || '').toLowerCase().trim();
        if (seenSigs.has(sig)) continue;
        seenSigs.add(sig);
        dedupedErrors.push(ve);
    }

    // Strip the per-element scratch fields we only needed for error
    // resolution — they aren't part of the planner-facing contract.
    for (const cap of out) {
        delete cap.describedby;
        delete cap.el_id;
    }

    return {
        elements: out,
        viewport: [window.innerWidth || 0, window.innerHeight || 0],
        validation_errors: dedupedErrors,
    };
}
"""


# ─── Helpers (Python-side) ────────────────────────────────────────────────────


def _build_ref(role: str, name: str, placeholder: str, bounds: tuple[int, int, int, int]) -> str:
    """
    Compute the content-addressed pre-ref. Same content → same ref.
    Bounds quantized to 8-px buckets (1px reflows don't change the ref).
    Returns the 10-char hex digest.
    """
    bucket = 8
    bx = (bounds[0] // bucket) * bucket
    by = (bounds[1] // bucket) * bucket
    bw = (bounds[2] // bucket) * bucket
    bh = (bounds[3] // bucket) * bucket
    key = f"{role}|{name}|{placeholder}|{bx},{by},{bw},{bh}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]


def _disambiguate_ref(base_ref: str, used_refs: dict[str, int]) -> str:
    """
    If `base_ref` already used, append `:N` (N starting at 2). DOM order
    determines suffixes (the JS pass walks document order).
    """
    if base_ref not in used_refs:
        used_refs[base_ref] = 1
        return base_ref
    used_refs[base_ref] += 1
    return f"{base_ref}:{used_refs[base_ref]}"


def _serialize_for_token_budget(elements: list[ElementInfo]) -> str:
    """
    Build the serialized form the planner prompt would see. We use a
    compact JSON representation — same shape the planner consumes.
    """
    return json.dumps([_to_planner_dict(e) for e in elements], separators=(",", ":"))


def _to_planner_dict(e: ElementInfo, *, drop_bounds: bool = False, drop_placeholder: bool = False) -> dict:
    """
    Compact dict for the planner prompt. Token-budget passes use the drop_*
    flags to shed weight.

    Placeholder is emitted only when it adds information beyond `name`.
    After Move 1's placeholder-as-name fallback, a fielded element where
    name was empty has name == placeholder; emitting both wastes tokens.
    """
    d: dict[str, Any] = {
        "ref": e.ref,
        "role": e.role,
        "name": e.name,
        "value": e.value,
    }
    if e.options:
        d["options"] = list(e.options)
    if (not drop_placeholder
            and e.placeholder
            and e.placeholder != e.name):
        d["placeholder"] = e.placeholder
    if not drop_bounds:
        d["bounds"] = list(e.bounds)
    if not e.visible:
        d["visible"] = False
    if not e.enabled:
        d["enabled"] = False
    if e.type and e.type != "text":  # "text" is the implied default
        d["type"] = e.type
    if e.role == "combobox" and e.autocomplete and not e.options:
        d["autocomplete"] = e.autocomplete
    if e.role == "combobox" or e.role == "option" or e.role == "listbox":
        logger.debug(
            f"[DOM] serialized for planner: ref={e.ref} role={e.role} "
            f"name={e.name!r} autocomplete={e.autocomplete!r} "
            f"options_count={len(e.options)} → emitted_keys={list(d.keys())}"
        )
    return d


# Each ElementInfo serializes to ~40 tokens at full fidelity, ~25 with
# bounds dropped, ~22 with placeholder dropped further. Heuristic constants
# tuned against tiktoken on representative form rows.
_TOKENS_PER_ELEMENT_FULL = 40
_TOKENS_PER_ELEMENT_NO_BOUNDS = 25
_TOKENS_PER_ELEMENT_NO_PLACEHOLDER = 22


def _estimate_tokens(elements: list[ElementInfo], *, drop_bounds: bool, drop_placeholder: bool) -> int:
    """
    Cheap token-cost estimate for budget enforcement. We don't call tiktoken
    on every read — that's milliseconds we don't need to spend. The constants
    are calibrated to be on the high side so we never under-budget.
    """
    if drop_placeholder:
        per = _TOKENS_PER_ELEMENT_NO_PLACEHOLDER
    elif drop_bounds:
        per = _TOKENS_PER_ELEMENT_NO_BOUNDS
    else:
        per = _TOKENS_PER_ELEMENT_FULL
    # Add roughly 1 token per option in select dropdowns
    extra = sum(len(e.options) for e in elements)
    return len(elements) * per + extra


def _apply_token_budget(
    elements: list[ElementInfo], budget: int
) -> tuple[list[ElementInfo], int, dict[str, bool]]:
    """
    Trim the element list to fit `budget` tokens (estimated). Returns
    (kept, truncated_count, flags) where `flags` records which budget
    passes fired (drop_bounds/drop_placeholder) so the serializer matches.

    Strategy in order:
      1. Try full fidelity.
      2. Drop bounds.
      3. Drop placeholders.
      4. Prune off-viewport elements.
      5. Prune from the tail.
    """
    flags = {"drop_bounds": False, "drop_placeholder": False}

    if _estimate_tokens(elements, drop_bounds=False, drop_placeholder=False) <= budget:
        return elements, 0, flags

    flags["drop_bounds"] = True
    if _estimate_tokens(elements, drop_bounds=True, drop_placeholder=False) <= budget:
        return elements, 0, flags

    flags["drop_placeholder"] = True
    if _estimate_tokens(elements, drop_bounds=True, drop_placeholder=True) <= budget:
        return elements, 0, flags

    # Step 4: drop invisible / off-viewport ones first (they're the
    # least likely to be the planner's target).
    visible = [e for e in elements if e.visible]
    invisible = [e for e in elements if not e.visible]
    truncated = len(elements) - len(visible)
    elements = visible
    if _estimate_tokens(elements, drop_bounds=True, drop_placeholder=True) <= budget:
        return elements, truncated, flags

    # Step 5: prune from the tail until we fit.
    while elements and _estimate_tokens(elements, drop_bounds=True, drop_placeholder=True) > budget:
        elements.pop()
        truncated += 1
    return elements, truncated, flags


# ─── Public entry point ───────────────────────────────────────────────────────


async def read_page_dom(
    page: Any,
    filter: FilterMode = "interactive",
    *,
    open_comboboxes: bool = False,
    use_cache: bool = True,
) -> PageDomTree:
    """
    Perceive the page's interactive element tree.

    Args:
      page                — Playwright `Page` (or any object exposing
                            `.evaluate(js, arg)` and `.locator(selector)`).
      filter              — "interactive" (default), "all", or "form".
      open_comboboxes     — reserved for the orchestrator's open-then-reperceive
                            flow; this perceiver returns combobox options ONLY
                            for native `<select>` elements regardless of this
                            flag (custom comboboxes need a click side-effect
                            that doesn't belong in the perceiver).
      use_cache           — reuse a recent cached tree when within TTL.

    Returns: `PageDomTree` with `elements`, `ref_to_locator`, `truncated`.
    Raises: nothing — failures degrade to an empty tree with a logged WARNING.
    """
    page_id = id(page)

    if use_cache:
        cached = _tree_cache.get(page_id)
        if cached is not None:
            ttl = float(getattr(config, "BROWSER_DOM_CACHE_TTL", 10.0))
            if (time.monotonic() - cached.read_at) < ttl:
                return cached

    # URL capture is best-effort. Playwright exposes `page.url` as a sync
    # property; test stubs may or may not provide it. Failures fall through
    # to "" — downstream nav-detection treats empty as "unknown".
    page_url = ""
    try:
        url_attr = getattr(page, "url", "")
        if isinstance(url_attr, str):
            page_url = url_attr
    except Exception:
        pass

    try:
        raw = await page.evaluate(_DOM_QUERY_JS, {"filter": filter, "openComboboxes": open_comboboxes})
    except Exception as e:
        # "Execution context was destroyed" / similar = navigation in flight.
        # The orchestrator treats this as a strong success signal AFTER a
        # submit click. We surface it via evaluate_failed rather than
        # silently returning an empty tree.
        logger.warning(f"[DOM] read_page_dom: page.evaluate failed ({type(e).__name__}: {e})")
        empty = PageDomTree(
            elements=[], ref_to_locator={}, truncated=0,
            read_at=time.monotonic(), viewport=(0, 0),
            evaluate_failed=True, url=page_url,
        )
        return empty

    if not isinstance(raw, dict) or "elements" not in raw:
        logger.warning(f"[DOM] read_page_dom: malformed JS return {type(raw).__name__}")
        empty = PageDomTree(
            elements=[], ref_to_locator={}, truncated=0,
            read_at=time.monotonic(), viewport=(0, 0),
            url=page_url,
        )
        return empty

    raw_list = raw.get("elements") or []
    viewport_raw = raw.get("viewport") or [0, 0]
    viewport = (int(viewport_raw[0] or 0), int(viewport_raw[1] or 0))

    used_refs: dict[str, int] = {}
    elements: list[ElementInfo] = []
    ref_to_locator: dict[str, Any] = {}
    idx_to_ref: dict[int, str] = {}  # map JS idx → ref for error anchoring

    for raw_el in raw_list:
        if not isinstance(raw_el, dict):
            continue
        try:
            idx = int(raw_el.get("idx", -1))
            tag = str(raw_el.get("tag", "") or "")
            role = str(raw_el.get("role", "") or "")
            name = str(raw_el.get("name", "") or "")
            placeholder = str(raw_el.get("placeholder", "") or "")
            value = str(raw_el.get("value", "") or "")
            options_raw = raw_el.get("options") or []
            options = tuple(str(o) for o in options_raw if isinstance(o, (str, int, float)))
            bounds_raw = raw_el.get("bounds") or [0, 0, 0, 0]
            bounds = tuple(int(b) for b in bounds_raw[:4])
            if len(bounds) != 4:
                continue
            visible = bool(raw_el.get("visible"))
            enabled = bool(raw_el.get("enabled"))
            type_ = str(raw_el.get("type", "") or "")
            form_id = str(raw_el.get("form_id", "") or "")
            in_dialog = bool(raw_el.get("in_dialog"))
            aria_invalid = bool(raw_el.get("aria_invalid"))
            autocomplete = str(raw_el.get("autocomplete", "") or "")
        except (ValueError, TypeError) as e:
            logger.debug(f"[DOM] skipping malformed element row: {e}")
            continue

        if idx < 0:
            continue

        # Placeholder-as-name fallback (Webflow/React forms commonly lack
        # <label> association but carry a placeholder that IS the visual
        # identifier the user reads). Apply at construction time so every
        # consumer — planner, debug logs, smoke output — sees the same
        # identifier. The original placeholder remains in `placeholder`
        # for token-budget passes that may want to drop it.
        if not name and placeholder:
            name = placeholder

        base_ref = _build_ref(role, name, placeholder, bounds)
        ref = _disambiguate_ref(base_ref, used_refs)

        # Locator via the data-tenka-idx attribute the JS just set.
        try:
            locator = page.locator(f"[data-tenka-idx='{idx}']")
        except Exception as e:
            logger.debug(f"[DOM] failed to build Locator for idx={idx}: {e}")
            continue

        if role == "combobox" or autocomplete:
            logger.debug(
                f"[DOM] combobox element idx={idx} tag={tag} role={role} "
                f"name={name!r} autocomplete={autocomplete!r} "
                f"options={options!r} form_id={form_id!r} value={value!r}"
            )

        elements.append(ElementInfo(
            ref=ref,
            role=role,
            name=name,
            placeholder=placeholder,
            value=value,
            options=options,
            bounds=bounds,
            visible=visible,
            enabled=enabled,
            type=type_,
            tag=tag,
            form_id=form_id,
            in_dialog=in_dialog,
            aria_invalid=aria_invalid,
            autocomplete=autocomplete,
        ))
        ref_to_locator[ref] = locator
        idx_to_ref[idx] = ref

    # Token-budget enforcement
    budget = int(getattr(config, "BROWSER_DOM_TREE_TOKEN_BUDGET", 4000))
    elements, truncated, _flags = _apply_token_budget(elements, budget)

    # Drop refs for pruned elements from the locator map (otherwise the
    # executor could try to act on a ref the planner never saw).
    kept_refs = {e.ref for e in elements}
    ref_to_locator = {r: loc for r, loc in ref_to_locator.items() if r in kept_refs}

    # Hydrate validation errors. Errors anchored to a pruned/unknown
    # idx fall through with field_ref="" — the orchestrator can still surface
    # them as a page-level alert message.
    raw_errors = raw.get("validation_errors") or []
    validation_errors_list: list[ValidationError] = []
    if isinstance(raw_errors, list):
        for raw_err in raw_errors:
            if not isinstance(raw_err, dict):
                continue
            try:
                fidx = int(raw_err.get("field_idx", -1))
                msg = str(raw_err.get("message", "") or "").strip()
                src = str(raw_err.get("source", "") or "").strip() or "error-class"
            except (ValueError, TypeError):
                continue
            if not msg:
                continue
            anchor_ref = idx_to_ref.get(fidx, "") if fidx >= 0 else ""
            # Drop errors anchored to a pruned ref — keeping them would risk
            # the orchestrator forwarding a feedback message that references
            # a field the planner can't see anymore.
            if fidx >= 0 and anchor_ref == "":
                continue
            validation_errors_list.append(ValidationError(
                field_ref=anchor_ref, message=msg[:300], source=src,
            ))

    tree = PageDomTree(
        elements=elements,
        ref_to_locator=ref_to_locator,
        truncated=truncated,
        read_at=time.monotonic(),
        viewport=viewport,
        validation_errors=tuple(validation_errors_list),
        url=page_url,
    )
    if use_cache:
        _tree_cache[page_id] = tree

    if truncated:
        logger.info(
            f"[DOM] read_page_dom: {len(elements)} kept, {truncated} pruned "
            f"to fit token budget (filter={filter})"
        )
    else:
        logger.debug(
            f"[DOM] read_page_dom: {len(elements)} elements (filter={filter}, viewport={viewport})"
        )

    if validation_errors_list or any(e.aria_invalid for e in elements):
        ai_fields = [
            f"{(e.name or '?')!r}(ref={e.ref})"
            for e in elements if e.aria_invalid
        ]
        logger.debug(
            f"[DOM] perception diag: aria_invalid fields="
            f"{ai_fields if ai_fields else '(none)'}, "
            f"validation_errors={len(validation_errors_list)}"
        )
        for i, ve in enumerate(validation_errors_list):
            field_name = "(page-level)" if not ve.field_ref else (
                f"ref={ve.field_ref}"
            )
            msg = (ve.message or "").replace("\n", " ").strip()[:160]
            logger.debug(
                f"[DOM] perception err[{i}] source={ve.source} "
                f"{field_name} msg={msg!r}"
            )

    return tree


def serialize_for_planner(tree: PageDomTree) -> str:
    """
    JSON-serialize the tree for inclusion in the DOM planner's prompt.
    Produces the same shape `_to_planner_dict` did, but with a
    deterministic ordering of fields. Keeps the tree behind a stable
    serialization API so callers don't poke into PageDomTree internals.
    """
    rows = [_to_planner_dict(e) for e in tree.elements]
    out: dict[str, Any] = {"elements": rows}
    if tree.truncated:
        out["_truncated"] = tree.truncated
    return json.dumps(out, separators=(",", ":"), ensure_ascii=False)
