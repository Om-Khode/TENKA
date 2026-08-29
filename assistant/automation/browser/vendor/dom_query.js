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
            el.dataset.droverIdx = String(idx);
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
        //    submit button), and `querySelector('[data-drover-idx]')` would
        //    arbitrarily pick the FIRST descendant in DOM order (which is
        //    the topmost form field, not the field that failed).
        //    Single-capture ancestors are still trustworthy — that's the
        //    "error sibling next to its input" case.
        if (fieldIdx < 0) {
            let cur = errEl.parentElement;
            let hops = 0;
            while (cur && cur !== document.body && hops < 6) {
                const captures = (cur.querySelectorAll
                    ? cur.querySelectorAll('[data-drover-idx]') : []);
                if (captures.length === 1) {
                    const i = parseInt(captures[0].dataset.droverIdx || '-1', 10);
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
        // Read in the same snapshot as the elements, deliberately. A
        // caller that fetched the url in a separate round trip could
        // compare a url from one moment against elements from another,
        // and navigation detection is exactly that comparison.
        url: (location && location.href) || '',
        validation_errors: dedupedErrors,
    };
}
