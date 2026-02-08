const ws = [
  { color: "red", primary: 600, secondary: 100 },
  { color: "green", primary: 600, secondary: 100 },
  { color: "blue", primary: 600, secondary: 100 },
  { color: "yellow", primary: 500, secondary: 100 },
  { color: "purple", primary: 600, secondary: 100 },
  { color: "teal", primary: 600, secondary: 100 },
  { color: "orange", primary: 600, secondary: 100 },
  { color: "cyan", primary: 600, secondary: 100 },
  { color: "lime", primary: 500, secondary: 100 },
  { color: "pink", primary: 600, secondary: 100 }
], qr = {
  inherit: "inherit",
  current: "currentColor",
  transparent: "transparent",
  black: "#000",
  white: "#fff",
  slate: {
    50: "#f8fafc",
    100: "#f1f5f9",
    200: "#e2e8f0",
    300: "#cbd5e1",
    400: "#94a3b8",
    500: "#64748b",
    600: "#475569",
    700: "#334155",
    800: "#1e293b",
    900: "#0f172a",
    950: "#020617"
  },
  gray: {
    50: "#f9fafb",
    100: "#f3f4f6",
    200: "#e5e7eb",
    300: "#d1d5db",
    400: "#9ca3af",
    500: "#6b7280",
    600: "#4b5563",
    700: "#374151",
    800: "#1f2937",
    900: "#111827",
    950: "#030712"
  },
  zinc: {
    50: "#fafafa",
    100: "#f4f4f5",
    200: "#e4e4e7",
    300: "#d4d4d8",
    400: "#a1a1aa",
    500: "#71717a",
    600: "#52525b",
    700: "#3f3f46",
    800: "#27272a",
    900: "#18181b",
    950: "#09090b"
  },
  neutral: {
    50: "#fafafa",
    100: "#f5f5f5",
    200: "#e5e5e5",
    300: "#d4d4d4",
    400: "#a3a3a3",
    500: "#737373",
    600: "#525252",
    700: "#404040",
    800: "#262626",
    900: "#171717",
    950: "#0a0a0a"
  },
  stone: {
    50: "#fafaf9",
    100: "#f5f5f4",
    200: "#e7e5e4",
    300: "#d6d3d1",
    400: "#a8a29e",
    500: "#78716c",
    600: "#57534e",
    700: "#44403c",
    800: "#292524",
    900: "#1c1917",
    950: "#0c0a09"
  },
  red: {
    50: "#fef2f2",
    100: "#fee2e2",
    200: "#fecaca",
    300: "#fca5a5",
    400: "#f87171",
    500: "#ef4444",
    600: "#dc2626",
    700: "#b91c1c",
    800: "#991b1b",
    900: "#7f1d1d",
    950: "#450a0a"
  },
  orange: {
    50: "#fff7ed",
    100: "#ffedd5",
    200: "#fed7aa",
    300: "#fdba74",
    400: "#fb923c",
    500: "#f97316",
    600: "#ea580c",
    700: "#c2410c",
    800: "#9a3412",
    900: "#7c2d12",
    950: "#431407"
  },
  amber: {
    50: "#fffbeb",
    100: "#fef3c7",
    200: "#fde68a",
    300: "#fcd34d",
    400: "#fbbf24",
    500: "#f59e0b",
    600: "#d97706",
    700: "#b45309",
    800: "#92400e",
    900: "#78350f",
    950: "#451a03"
  },
  yellow: {
    50: "#fefce8",
    100: "#fef9c3",
    200: "#fef08a",
    300: "#fde047",
    400: "#facc15",
    500: "#eab308",
    600: "#ca8a04",
    700: "#a16207",
    800: "#854d0e",
    900: "#713f12",
    950: "#422006"
  },
  lime: {
    50: "#f7fee7",
    100: "#ecfccb",
    200: "#d9f99d",
    300: "#bef264",
    400: "#a3e635",
    500: "#84cc16",
    600: "#65a30d",
    700: "#4d7c0f",
    800: "#3f6212",
    900: "#365314",
    950: "#1a2e05"
  },
  green: {
    50: "#f0fdf4",
    100: "#dcfce7",
    200: "#bbf7d0",
    300: "#86efac",
    400: "#4ade80",
    500: "#22c55e",
    600: "#16a34a",
    700: "#15803d",
    800: "#166534",
    900: "#14532d",
    950: "#052e16"
  },
  emerald: {
    50: "#ecfdf5",
    100: "#d1fae5",
    200: "#a7f3d0",
    300: "#6ee7b7",
    400: "#34d399",
    500: "#10b981",
    600: "#059669",
    700: "#047857",
    800: "#065f46",
    900: "#064e3b",
    950: "#022c22"
  },
  teal: {
    50: "#f0fdfa",
    100: "#ccfbf1",
    200: "#99f6e4",
    300: "#5eead4",
    400: "#2dd4bf",
    500: "#14b8a6",
    600: "#0d9488",
    700: "#0f766e",
    800: "#115e59",
    900: "#134e4a",
    950: "#042f2e"
  },
  cyan: {
    50: "#ecfeff",
    100: "#cffafe",
    200: "#a5f3fc",
    300: "#67e8f9",
    400: "#22d3ee",
    500: "#06b6d4",
    600: "#0891b2",
    700: "#0e7490",
    800: "#155e75",
    900: "#164e63",
    950: "#083344"
  },
  sky: {
    50: "#f0f9ff",
    100: "#e0f2fe",
    200: "#bae6fd",
    300: "#7dd3fc",
    400: "#38bdf8",
    500: "#0ea5e9",
    600: "#0284c7",
    700: "#0369a1",
    800: "#075985",
    900: "#0c4a6e",
    950: "#082f49"
  },
  blue: {
    50: "#eff6ff",
    100: "#dbeafe",
    200: "#bfdbfe",
    300: "#93c5fd",
    400: "#60a5fa",
    500: "#3b82f6",
    600: "#2563eb",
    700: "#1d4ed8",
    800: "#1e40af",
    900: "#1e3a8a",
    950: "#172554"
  },
  indigo: {
    50: "#eef2ff",
    100: "#e0e7ff",
    200: "#c7d2fe",
    300: "#a5b4fc",
    400: "#818cf8",
    500: "#6366f1",
    600: "#4f46e5",
    700: "#4338ca",
    800: "#3730a3",
    900: "#312e81",
    950: "#1e1b4b"
  },
  violet: {
    50: "#f5f3ff",
    100: "#ede9fe",
    200: "#ddd6fe",
    300: "#c4b5fd",
    400: "#a78bfa",
    500: "#8b5cf6",
    600: "#7c3aed",
    700: "#6d28d9",
    800: "#5b21b6",
    900: "#4c1d95",
    950: "#2e1065"
  },
  purple: {
    50: "#faf5ff",
    100: "#f3e8ff",
    200: "#e9d5ff",
    300: "#d8b4fe",
    400: "#c084fc",
    500: "#a855f7",
    600: "#9333ea",
    700: "#7e22ce",
    800: "#6b21a8",
    900: "#581c87",
    950: "#3b0764"
  },
  fuchsia: {
    50: "#fdf4ff",
    100: "#fae8ff",
    200: "#f5d0fe",
    300: "#f0abfc",
    400: "#e879f9",
    500: "#d946ef",
    600: "#c026d3",
    700: "#a21caf",
    800: "#86198f",
    900: "#701a75",
    950: "#4a044e"
  },
  pink: {
    50: "#fdf2f8",
    100: "#fce7f3",
    200: "#fbcfe8",
    300: "#f9a8d4",
    400: "#f472b6",
    500: "#ec4899",
    600: "#db2777",
    700: "#be185d",
    800: "#9d174d",
    900: "#831843",
    950: "#500724"
  },
  rose: {
    50: "#fff1f2",
    100: "#ffe4e6",
    200: "#fecdd3",
    300: "#fda4af",
    400: "#fb7185",
    500: "#f43f5e",
    600: "#e11d48",
    700: "#be123c",
    800: "#9f1239",
    900: "#881337",
    950: "#4c0519"
  }
};
ws.reduce(
  (e, { color: t, primary: n, secondary: r }) => ({
    ...e,
    [t]: {
      primary: qr[t][n],
      secondary: qr[t][r]
    }
  }),
  {}
);
function vs(e) {
  let t, n = e[0], r = 1;
  for (; r < e.length; ) {
    const i = e[r], l = e[r + 1];
    if (r += 2, (i === "optionalAccess" || i === "optionalCall") && n == null)
      return;
    i === "access" || i === "optionalAccess" ? (t = n, n = l(n)) : (i === "call" || i === "optionalCall") && (n = l((...s) => n.call(t, ...s)), t = void 0);
  }
  return n;
}
class sn extends Error {
  constructor(t) {
    super(t), this.name = "ShareError";
  }
}
async function ys(e, t) {
  if (window.__gradio_space__ == null)
    throw new sn("Must be on Spaces to share.");
  let n, r, i;
  if (t === "url") {
    const o = await fetch(e);
    n = await o.blob(), r = o.headers.get("content-type") || "", i = o.headers.get("content-disposition") || "";
  } else
    n = Es(e), r = e.split(";")[0].split(":")[1], i = "file" + r.split("/")[1];
  const l = new File([n], i, { type: r }), s = await fetch("https://huggingface.co/uploads", {
    method: "POST",
    body: l,
    headers: {
      "Content-Type": l.type,
      "X-Requested-With": "XMLHttpRequest"
    }
  });
  if (!s.ok) {
    if (vs([s, "access", (o) => o.headers, "access", (o) => o.get, "call", (o) => o("content-type"), "optionalAccess", (o) => o.includes, "call", (o) => o("application/json")])) {
      const o = await s.json();
      throw new sn(`Upload failed: ${o.error}`);
    }
    throw new sn("Upload failed.");
  }
  return await s.text();
}
function Es(e) {
  for (var t = e.split(","), n = t[0].match(/:(.*?);/)[1], r = atob(t[1]), i = r.length, l = new Uint8Array(i); i--; )
    l[i] = r.charCodeAt(i);
  return new Blob([l], { type: n });
}
const {
  SvelteComponent: Ss,
  assign: ks,
  create_slot: Ts,
  detach: Bs,
  element: As,
  get_all_dirty_from_scope: Cs,
  get_slot_changes: Hs,
  get_spread_update: Ps,
  init: Ns,
  insert: Is,
  safe_not_equal: Ls,
  set_dynamic_element_data: xr,
  set_style: ue,
  toggle_class: xe,
  transition_in: bl,
  transition_out: wl,
  update_slot_base: Os
} = window.__gradio__svelte__internal;
function Ms(e) {
  let t, n, r;
  const i = (
    /*#slots*/
    e[17].default
  ), l = Ts(
    i,
    e,
    /*$$scope*/
    e[16],
    null
  );
  let s = [
    { "data-testid": (
      /*test_id*/
      e[7]
    ) },
    { id: (
      /*elem_id*/
      e[2]
    ) },
    {
      class: n = "block " + /*elem_classes*/
      e[3].join(" ") + " svelte-1t38q2d"
    }
  ], a = {};
  for (let o = 0; o < s.length; o += 1)
    a = ks(a, s[o]);
  return {
    c() {
      t = As(
        /*tag*/
        e[14]
      ), l && l.c(), xr(
        /*tag*/
        e[14]
      )(t, a), xe(
        t,
        "hidden",
        /*visible*/
        e[10] === !1
      ), xe(
        t,
        "padded",
        /*padding*/
        e[6]
      ), xe(
        t,
        "border_focus",
        /*border_mode*/
        e[5] === "focus"
      ), xe(t, "hide-container", !/*explicit_call*/
      e[8] && !/*container*/
      e[9]), ue(t, "height", typeof /*height*/
      e[0] == "number" ? (
        /*height*/
        e[0] + "px"
      ) : void 0), ue(t, "width", typeof /*width*/
      e[1] == "number" ? `calc(min(${/*width*/
      e[1]}px, 100%))` : void 0), ue(
        t,
        "border-style",
        /*variant*/
        e[4]
      ), ue(
        t,
        "overflow",
        /*allow_overflow*/
        e[11] ? "visible" : "hidden"
      ), ue(
        t,
        "flex-grow",
        /*scale*/
        e[12]
      ), ue(t, "min-width", `calc(min(${/*min_width*/
      e[13]}px, 100%))`), ue(t, "border-width", "var(--block-border-width)");
    },
    m(o, u) {
      Is(o, t, u), l && l.m(t, null), r = !0;
    },
    p(o, u) {
      l && l.p && (!r || u & /*$$scope*/
      65536) && Os(
        l,
        i,
        o,
        /*$$scope*/
        o[16],
        r ? Hs(
          i,
          /*$$scope*/
          o[16],
          u,
          null
        ) : Cs(
          /*$$scope*/
          o[16]
        ),
        null
      ), xr(
        /*tag*/
        o[14]
      )(t, a = Ps(s, [
        (!r || u & /*test_id*/
        128) && { "data-testid": (
          /*test_id*/
          o[7]
        ) },
        (!r || u & /*elem_id*/
        4) && { id: (
          /*elem_id*/
          o[2]
        ) },
        (!r || u & /*elem_classes*/
        8 && n !== (n = "block " + /*elem_classes*/
        o[3].join(" ") + " svelte-1t38q2d")) && { class: n }
      ])), xe(
        t,
        "hidden",
        /*visible*/
        o[10] === !1
      ), xe(
        t,
        "padded",
        /*padding*/
        o[6]
      ), xe(
        t,
        "border_focus",
        /*border_mode*/
        o[5] === "focus"
      ), xe(t, "hide-container", !/*explicit_call*/
      o[8] && !/*container*/
      o[9]), u & /*height*/
      1 && ue(t, "height", typeof /*height*/
      o[0] == "number" ? (
        /*height*/
        o[0] + "px"
      ) : void 0), u & /*width*/
      2 && ue(t, "width", typeof /*width*/
      o[1] == "number" ? `calc(min(${/*width*/
      o[1]}px, 100%))` : void 0), u & /*variant*/
      16 && ue(
        t,
        "border-style",
        /*variant*/
        o[4]
      ), u & /*allow_overflow*/
      2048 && ue(
        t,
        "overflow",
        /*allow_overflow*/
        o[11] ? "visible" : "hidden"
      ), u & /*scale*/
      4096 && ue(
        t,
        "flex-grow",
        /*scale*/
        o[12]
      ), u & /*min_width*/
      8192 && ue(t, "min-width", `calc(min(${/*min_width*/
      o[13]}px, 100%))`);
    },
    i(o) {
      r || (bl(l, o), r = !0);
    },
    o(o) {
      wl(l, o), r = !1;
    },
    d(o) {
      o && Bs(t), l && l.d(o);
    }
  };
}
function Rs(e) {
  let t, n = (
    /*tag*/
    e[14] && Ms(e)
  );
  return {
    c() {
      n && n.c();
    },
    m(r, i) {
      n && n.m(r, i), t = !0;
    },
    p(r, [i]) {
      /*tag*/
      r[14] && n.p(r, i);
    },
    i(r) {
      t || (bl(n, r), t = !0);
    },
    o(r) {
      wl(n, r), t = !1;
    },
    d(r) {
      n && n.d(r);
    }
  };
}
function Ds(e, t, n) {
  let { $$slots: r = {}, $$scope: i } = t, { height: l = void 0 } = t, { width: s = void 0 } = t, { elem_id: a = "" } = t, { elem_classes: o = [] } = t, { variant: u = "solid" } = t, { border_mode: f = "base" } = t, { padding: c = !0 } = t, { type: h = "normal" } = t, { test_id: _ = void 0 } = t, { explicit_call: d = !1 } = t, { container: g = !0 } = t, { visible: S = !0 } = t, { allow_overflow: b = !0 } = t, { scale: T = null } = t, { min_width: y = 0 } = t, m = h === "fieldset" ? "fieldset" : "div";
  return e.$$set = (v) => {
    "height" in v && n(0, l = v.height), "width" in v && n(1, s = v.width), "elem_id" in v && n(2, a = v.elem_id), "elem_classes" in v && n(3, o = v.elem_classes), "variant" in v && n(4, u = v.variant), "border_mode" in v && n(5, f = v.border_mode), "padding" in v && n(6, c = v.padding), "type" in v && n(15, h = v.type), "test_id" in v && n(7, _ = v.test_id), "explicit_call" in v && n(8, d = v.explicit_call), "container" in v && n(9, g = v.container), "visible" in v && n(10, S = v.visible), "allow_overflow" in v && n(11, b = v.allow_overflow), "scale" in v && n(12, T = v.scale), "min_width" in v && n(13, y = v.min_width), "$$scope" in v && n(16, i = v.$$scope);
  }, [
    l,
    s,
    a,
    o,
    u,
    f,
    c,
    _,
    d,
    g,
    S,
    b,
    T,
    y,
    m,
    h,
    i,
    r
  ];
}
class vl extends Ss {
  constructor(t) {
    super(), Ns(this, t, Ds, Rs, Ls, {
      height: 0,
      width: 1,
      elem_id: 2,
      elem_classes: 3,
      variant: 4,
      border_mode: 5,
      padding: 6,
      type: 15,
      test_id: 7,
      explicit_call: 8,
      container: 9,
      visible: 10,
      allow_overflow: 11,
      scale: 12,
      min_width: 13
    });
  }
}
const {
  SvelteComponent: Us,
  append: Cn,
  attr: Wt,
  create_component: Fs,
  destroy_component: Gs,
  detach: qs,
  element: jr,
  init: xs,
  insert: js,
  mount_component: zs,
  safe_not_equal: Vs,
  set_data: Xs,
  space: Zs,
  text: Ws,
  toggle_class: je,
  transition_in: Js,
  transition_out: Ys
} = window.__gradio__svelte__internal;
function Qs(e) {
  let t, n, r, i, l, s;
  return r = new /*Icon*/
  e[1]({}), {
    c() {
      t = jr("label"), n = jr("span"), Fs(r.$$.fragment), i = Zs(), l = Ws(
        /*label*/
        e[0]
      ), Wt(n, "class", "svelte-9gxdi0"), Wt(t, "for", ""), Wt(t, "data-testid", "block-label"), Wt(t, "class", "svelte-9gxdi0"), je(t, "hide", !/*show_label*/
      e[2]), je(t, "sr-only", !/*show_label*/
      e[2]), je(
        t,
        "float",
        /*float*/
        e[4]
      ), je(
        t,
        "hide-label",
        /*disable*/
        e[3]
      );
    },
    m(a, o) {
      js(a, t, o), Cn(t, n), zs(r, n, null), Cn(t, i), Cn(t, l), s = !0;
    },
    p(a, [o]) {
      (!s || o & /*label*/
      1) && Xs(
        l,
        /*label*/
        a[0]
      ), (!s || o & /*show_label*/
      4) && je(t, "hide", !/*show_label*/
      a[2]), (!s || o & /*show_label*/
      4) && je(t, "sr-only", !/*show_label*/
      a[2]), (!s || o & /*float*/
      16) && je(
        t,
        "float",
        /*float*/
        a[4]
      ), (!s || o & /*disable*/
      8) && je(
        t,
        "hide-label",
        /*disable*/
        a[3]
      );
    },
    i(a) {
      s || (Js(r.$$.fragment, a), s = !0);
    },
    o(a) {
      Ys(r.$$.fragment, a), s = !1;
    },
    d(a) {
      a && qs(t), Gs(r);
    }
  };
}
function Ks(e, t, n) {
  let { label: r = null } = t, { Icon: i } = t, { show_label: l = !0 } = t, { disable: s = !1 } = t, { float: a = !0 } = t;
  return e.$$set = (o) => {
    "label" in o && n(0, r = o.label), "Icon" in o && n(1, i = o.Icon), "show_label" in o && n(2, l = o.show_label), "disable" in o && n(3, s = o.disable), "float" in o && n(4, a = o.float);
  }, [r, i, l, s, a];
}
class yl extends Us {
  constructor(t) {
    super(), xs(this, t, Ks, Qs, Vs, {
      label: 0,
      Icon: 1,
      show_label: 2,
      disable: 3,
      float: 4
    });
  }
}
const {
  SvelteComponent: $s,
  append: fr,
  attr: Me,
  bubble: eo,
  create_component: to,
  destroy_component: no,
  detach: El,
  element: cr,
  init: ro,
  insert: Sl,
  listen: io,
  mount_component: lo,
  safe_not_equal: so,
  set_data: oo,
  set_style: Jt,
  space: ao,
  text: uo,
  toggle_class: ge,
  transition_in: fo,
  transition_out: co
} = window.__gradio__svelte__internal;
function zr(e) {
  let t, n;
  return {
    c() {
      t = cr("span"), n = uo(
        /*label*/
        e[1]
      ), Me(t, "class", "svelte-lpi64a");
    },
    m(r, i) {
      Sl(r, t, i), fr(t, n);
    },
    p(r, i) {
      i & /*label*/
      2 && oo(
        n,
        /*label*/
        r[1]
      );
    },
    d(r) {
      r && El(t);
    }
  };
}
function ho(e) {
  let t, n, r, i, l, s, a, o = (
    /*show_label*/
    e[2] && zr(e)
  );
  return i = new /*Icon*/
  e[0]({}), {
    c() {
      t = cr("button"), o && o.c(), n = ao(), r = cr("div"), to(i.$$.fragment), Me(r, "class", "svelte-lpi64a"), ge(
        r,
        "small",
        /*size*/
        e[4] === "small"
      ), ge(
        r,
        "large",
        /*size*/
        e[4] === "large"
      ), t.disabled = /*disabled*/
      e[7], Me(
        t,
        "aria-label",
        /*label*/
        e[1]
      ), Me(
        t,
        "aria-haspopup",
        /*hasPopup*/
        e[8]
      ), Me(
        t,
        "title",
        /*label*/
        e[1]
      ), Me(t, "class", "svelte-lpi64a"), ge(
        t,
        "pending",
        /*pending*/
        e[3]
      ), ge(
        t,
        "padded",
        /*padded*/
        e[5]
      ), ge(
        t,
        "highlight",
        /*highlight*/
        e[6]
      ), ge(
        t,
        "transparent",
        /*transparent*/
        e[9]
      ), Jt(t, "color", !/*disabled*/
      e[7] && /*_color*/
      e[11] ? (
        /*_color*/
        e[11]
      ) : "var(--block-label-text-color)"), Jt(t, "--bg-color", /*disabled*/
      e[7] ? "auto" : (
        /*background*/
        e[10]
      ));
    },
    m(u, f) {
      Sl(u, t, f), o && o.m(t, null), fr(t, n), fr(t, r), lo(i, r, null), l = !0, s || (a = io(
        t,
        "click",
        /*click_handler*/
        e[13]
      ), s = !0);
    },
    p(u, [f]) {
      /*show_label*/
      u[2] ? o ? o.p(u, f) : (o = zr(u), o.c(), o.m(t, n)) : o && (o.d(1), o = null), (!l || f & /*size*/
      16) && ge(
        r,
        "small",
        /*size*/
        u[4] === "small"
      ), (!l || f & /*size*/
      16) && ge(
        r,
        "large",
        /*size*/
        u[4] === "large"
      ), (!l || f & /*disabled*/
      128) && (t.disabled = /*disabled*/
      u[7]), (!l || f & /*label*/
      2) && Me(
        t,
        "aria-label",
        /*label*/
        u[1]
      ), (!l || f & /*hasPopup*/
      256) && Me(
        t,
        "aria-haspopup",
        /*hasPopup*/
        u[8]
      ), (!l || f & /*label*/
      2) && Me(
        t,
        "title",
        /*label*/
        u[1]
      ), (!l || f & /*pending*/
      8) && ge(
        t,
        "pending",
        /*pending*/
        u[3]
      ), (!l || f & /*padded*/
      32) && ge(
        t,
        "padded",
        /*padded*/
        u[5]
      ), (!l || f & /*highlight*/
      64) && ge(
        t,
        "highlight",
        /*highlight*/
        u[6]
      ), (!l || f & /*transparent*/
      512) && ge(
        t,
        "transparent",
        /*transparent*/
        u[9]
      ), f & /*disabled, _color*/
      2176 && Jt(t, "color", !/*disabled*/
      u[7] && /*_color*/
      u[11] ? (
        /*_color*/
        u[11]
      ) : "var(--block-label-text-color)"), f & /*disabled, background*/
      1152 && Jt(t, "--bg-color", /*disabled*/
      u[7] ? "auto" : (
        /*background*/
        u[10]
      ));
    },
    i(u) {
      l || (fo(i.$$.fragment, u), l = !0);
    },
    o(u) {
      co(i.$$.fragment, u), l = !1;
    },
    d(u) {
      u && El(t), o && o.d(), no(i), s = !1, a();
    }
  };
}
function _o(e, t, n) {
  let r, { Icon: i } = t, { label: l = "" } = t, { show_label: s = !1 } = t, { pending: a = !1 } = t, { size: o = "small" } = t, { padded: u = !0 } = t, { highlight: f = !1 } = t, { disabled: c = !1 } = t, { hasPopup: h = !1 } = t, { color: _ = "var(--block-label-text-color)" } = t, { transparent: d = !1 } = t, { background: g = "var(--background-fill-primary)" } = t;
  function S(b) {
    eo.call(this, e, b);
  }
  return e.$$set = (b) => {
    "Icon" in b && n(0, i = b.Icon), "label" in b && n(1, l = b.label), "show_label" in b && n(2, s = b.show_label), "pending" in b && n(3, a = b.pending), "size" in b && n(4, o = b.size), "padded" in b && n(5, u = b.padded), "highlight" in b && n(6, f = b.highlight), "disabled" in b && n(7, c = b.disabled), "hasPopup" in b && n(8, h = b.hasPopup), "color" in b && n(12, _ = b.color), "transparent" in b && n(9, d = b.transparent), "background" in b && n(10, g = b.background);
  }, e.$$.update = () => {
    e.$$.dirty & /*highlight, color*/
    4160 && n(11, r = f ? "var(--color-accent)" : _);
  }, [
    i,
    l,
    s,
    a,
    o,
    u,
    f,
    c,
    h,
    d,
    g,
    r,
    _,
    S
  ];
}
class ht extends $s {
  constructor(t) {
    super(), ro(this, t, _o, ho, so, {
      Icon: 0,
      label: 1,
      show_label: 2,
      pending: 3,
      size: 4,
      padded: 5,
      highlight: 6,
      disabled: 7,
      hasPopup: 8,
      color: 12,
      transparent: 9,
      background: 10
    });
  }
}
const {
  SvelteComponent: mo,
  append: go,
  attr: Hn,
  binding_callbacks: po,
  create_slot: bo,
  detach: wo,
  element: Vr,
  get_all_dirty_from_scope: vo,
  get_slot_changes: yo,
  init: Eo,
  insert: So,
  safe_not_equal: ko,
  toggle_class: ze,
  transition_in: To,
  transition_out: Bo,
  update_slot_base: Ao
} = window.__gradio__svelte__internal;
function Co(e) {
  let t, n, r;
  const i = (
    /*#slots*/
    e[5].default
  ), l = bo(
    i,
    e,
    /*$$scope*/
    e[4],
    null
  );
  return {
    c() {
      t = Vr("div"), n = Vr("div"), l && l.c(), Hn(n, "class", "icon svelte-3w3rth"), Hn(t, "class", "empty svelte-3w3rth"), Hn(t, "aria-label", "Empty value"), ze(
        t,
        "small",
        /*size*/
        e[0] === "small"
      ), ze(
        t,
        "large",
        /*size*/
        e[0] === "large"
      ), ze(
        t,
        "unpadded_box",
        /*unpadded_box*/
        e[1]
      ), ze(
        t,
        "small_parent",
        /*parent_height*/
        e[3]
      );
    },
    m(s, a) {
      So(s, t, a), go(t, n), l && l.m(n, null), e[6](t), r = !0;
    },
    p(s, [a]) {
      l && l.p && (!r || a & /*$$scope*/
      16) && Ao(
        l,
        i,
        s,
        /*$$scope*/
        s[4],
        r ? yo(
          i,
          /*$$scope*/
          s[4],
          a,
          null
        ) : vo(
          /*$$scope*/
          s[4]
        ),
        null
      ), (!r || a & /*size*/
      1) && ze(
        t,
        "small",
        /*size*/
        s[0] === "small"
      ), (!r || a & /*size*/
      1) && ze(
        t,
        "large",
        /*size*/
        s[0] === "large"
      ), (!r || a & /*unpadded_box*/
      2) && ze(
        t,
        "unpadded_box",
        /*unpadded_box*/
        s[1]
      ), (!r || a & /*parent_height*/
      8) && ze(
        t,
        "small_parent",
        /*parent_height*/
        s[3]
      );
    },
    i(s) {
      r || (To(l, s), r = !0);
    },
    o(s) {
      Bo(l, s), r = !1;
    },
    d(s) {
      s && wo(t), l && l.d(s), e[6](null);
    }
  };
}
function Ho(e) {
  let t, n = e[0], r = 1;
  for (; r < e.length; ) {
    const i = e[r], l = e[r + 1];
    if (r += 2, (i === "optionalAccess" || i === "optionalCall") && n == null)
      return;
    i === "access" || i === "optionalAccess" ? (t = n, n = l(n)) : (i === "call" || i === "optionalCall") && (n = l((...s) => n.call(t, ...s)), t = void 0);
  }
  return n;
}
function Po(e, t, n) {
  let r, { $$slots: i = {}, $$scope: l } = t, { size: s = "small" } = t, { unpadded_box: a = !1 } = t, o;
  function u(c) {
    if (!c)
      return !1;
    const { height: h } = c.getBoundingClientRect(), { height: _ } = Ho([
      c,
      "access",
      (d) => d.parentElement,
      "optionalAccess",
      (d) => d.getBoundingClientRect,
      "call",
      (d) => d()
    ]) || { height: h };
    return h > _ + 2;
  }
  function f(c) {
    po[c ? "unshift" : "push"](() => {
      o = c, n(2, o);
    });
  }
  return e.$$set = (c) => {
    "size" in c && n(0, s = c.size), "unpadded_box" in c && n(1, a = c.unpadded_box), "$$scope" in c && n(4, l = c.$$scope);
  }, e.$$.update = () => {
    e.$$.dirty & /*el*/
    4 && n(3, r = u(o));
  }, [s, a, o, r, l, i, f];
}
class kl extends mo {
  constructor(t) {
    super(), Eo(this, t, Po, Co, ko, { size: 0, unpadded_box: 1 });
  }
}
const {
  SvelteComponent: No,
  append: Pn,
  attr: Be,
  detach: Io,
  init: Lo,
  insert: Oo,
  noop: Nn,
  safe_not_equal: Mo,
  set_style: Ne,
  svg_element: Yt
} = window.__gradio__svelte__internal;
function Ro(e) {
  let t, n, r, i;
  return {
    c() {
      t = Yt("svg"), n = Yt("g"), r = Yt("path"), i = Yt("path"), Be(r, "d", "M18,6L6.087,17.913"), Ne(r, "fill", "none"), Ne(r, "fill-rule", "nonzero"), Ne(r, "stroke-width", "2px"), Be(n, "transform", "matrix(1.14096,-0.140958,-0.140958,1.14096,-0.0559523,0.0559523)"), Be(i, "d", "M4.364,4.364L19.636,19.636"), Ne(i, "fill", "none"), Ne(i, "fill-rule", "nonzero"), Ne(i, "stroke-width", "2px"), Be(t, "width", "100%"), Be(t, "height", "100%"), Be(t, "viewBox", "0 0 24 24"), Be(t, "version", "1.1"), Be(t, "xmlns", "http://www.w3.org/2000/svg"), Be(t, "xmlns:xlink", "http://www.w3.org/1999/xlink"), Be(t, "xml:space", "preserve"), Be(t, "stroke", "currentColor"), Ne(t, "fill-rule", "evenodd"), Ne(t, "clip-rule", "evenodd"), Ne(t, "stroke-linecap", "round"), Ne(t, "stroke-linejoin", "round");
    },
    m(l, s) {
      Oo(l, t, s), Pn(t, n), Pn(n, r), Pn(t, i);
    },
    p: Nn,
    i: Nn,
    o: Nn,
    d(l) {
      l && Io(t);
    }
  };
}
class Do extends No {
  constructor(t) {
    super(), Lo(this, t, null, Ro, Mo, {});
  }
}
const {
  SvelteComponent: Uo,
  append: Fo,
  attr: Nt,
  detach: Go,
  init: qo,
  insert: xo,
  noop: In,
  safe_not_equal: jo,
  svg_element: Xr
} = window.__gradio__svelte__internal;
function zo(e) {
  let t, n;
  return {
    c() {
      t = Xr("svg"), n = Xr("path"), Nt(n, "d", "M23,20a5,5,0,0,0-3.89,1.89L11.8,17.32a4.46,4.46,0,0,0,0-2.64l7.31-4.57A5,5,0,1,0,18,7a4.79,4.79,0,0,0,.2,1.32l-7.31,4.57a5,5,0,1,0,0,6.22l7.31,4.57A4.79,4.79,0,0,0,18,25a5,5,0,1,0,5-5ZM23,4a3,3,0,1,1-3,3A3,3,0,0,1,23,4ZM7,19a3,3,0,1,1,3-3A3,3,0,0,1,7,19Zm16,9a3,3,0,1,1,3-3A3,3,0,0,1,23,28Z"), Nt(n, "fill", "currentColor"), Nt(t, "id", "icon"), Nt(t, "xmlns", "http://www.w3.org/2000/svg"), Nt(t, "viewBox", "0 0 32 32");
    },
    m(r, i) {
      xo(r, t, i), Fo(t, n);
    },
    p: In,
    i: In,
    o: In,
    d(r) {
      r && Go(t);
    }
  };
}
class Vo extends Uo {
  constructor(t) {
    super(), qo(this, t, null, zo, jo, {});
  }
}
const {
  SvelteComponent: Xo,
  append: Zo,
  attr: lt,
  detach: Wo,
  init: Jo,
  insert: Yo,
  noop: Ln,
  safe_not_equal: Qo,
  svg_element: Zr
} = window.__gradio__svelte__internal;
function Ko(e) {
  let t, n;
  return {
    c() {
      t = Zr("svg"), n = Zr("path"), lt(n, "fill", "currentColor"), lt(n, "d", "M26 24v4H6v-4H4v4a2 2 0 0 0 2 2h20a2 2 0 0 0 2-2v-4zm0-10l-1.41-1.41L17 20.17V2h-2v18.17l-7.59-7.58L6 14l10 10l10-10z"), lt(t, "xmlns", "http://www.w3.org/2000/svg"), lt(t, "width", "100%"), lt(t, "height", "100%"), lt(t, "viewBox", "0 0 32 32");
    },
    m(r, i) {
      Yo(r, t, i), Zo(t, n);
    },
    p: Ln,
    i: Ln,
    o: Ln,
    d(r) {
      r && Wo(t);
    }
  };
}
class $o extends Xo {
  constructor(t) {
    super(), Jo(this, t, null, Ko, Qo, {});
  }
}
const {
  SvelteComponent: ea,
  append: On,
  attr: Ae,
  detach: ta,
  init: na,
  insert: ra,
  noop: Mn,
  safe_not_equal: ia,
  svg_element: Qt
} = window.__gradio__svelte__internal;
function la(e) {
  let t, n, r, i;
  return {
    c() {
      t = Qt("svg"), n = Qt("g"), r = Qt("path"), i = Qt("path"), Ae(r, "fill", "currentColor"), Ae(r, "d", "m5.505 11.41l.53.53l-.53-.53ZM3 14.952h-.75H3ZM9.048 21v.75V21ZM11.41 5.505l-.53-.53l.53.53Zm1.831 12.34a.75.75 0 0 0 1.06-1.061l-1.06 1.06ZM7.216 9.697a.75.75 0 1 0-1.06 1.061l1.06-1.06Zm10.749 2.362l-5.905 5.905l1.06 1.06l5.905-5.904l-1.06-1.06Zm-11.93-.12l5.905-5.905l-1.06-1.06l-5.905 5.904l1.06 1.06Zm0 6.025c-.85-.85-1.433-1.436-1.812-1.933c-.367-.481-.473-.79-.473-1.08h-1.5c0 .749.312 1.375.78 1.99c.455.596 1.125 1.263 1.945 2.083l1.06-1.06Zm-1.06-7.086c-.82.82-1.49 1.488-1.945 2.084c-.468.614-.78 1.24-.78 1.99h1.5c0-.29.106-.6.473-1.08c.38-.498.962-1.083 1.812-1.933l-1.06-1.06Zm7.085 7.086c-.85.85-1.435 1.433-1.933 1.813c-.48.366-.79.472-1.08.472v1.5c.75 0 1.376-.312 1.99-.78c.596-.455 1.264-1.125 2.084-1.945l-1.06-1.06Zm-7.085 1.06c.82.82 1.487 1.49 2.084 1.945c.614.468 1.24.78 1.989.78v-1.5c-.29 0-.599-.106-1.08-.473c-.497-.38-1.083-.962-1.933-1.812l-1.06 1.06Zm12.99-12.99c.85.85 1.433 1.436 1.813 1.933c.366.481.472.79.472 1.08h1.5c0-.749-.312-1.375-.78-1.99c-.455-.596-1.125-1.263-1.945-2.083l-1.06 1.06Zm1.06 7.086c.82-.82 1.49-1.488 1.945-2.084c.468-.614.78-1.24.78-1.99h-1.5c0 .29-.106.6-.473 1.08c-.38.498-.962 1.083-1.812 1.933l1.06 1.06Zm0-8.146c-.82-.82-1.487-1.49-2.084-1.945c-.614-.468-1.24-.78-1.989-.78v1.5c.29 0 .599.106 1.08.473c.497.38 1.083.962 1.933 1.812l1.06-1.06Zm-7.085 1.06c.85-.85 1.435-1.433 1.933-1.812c.48-.367.79-.473 1.08-.473v-1.5c-.75 0-1.376.312-1.99.78c-.596.455-1.264 1.125-2.084 1.945l1.06 1.06Zm2.362 10.749L7.216 9.698l-1.06 1.061l7.085 7.085l1.06-1.06Z"), Ae(i, "stroke", "currentColor"), Ae(i, "stroke-linecap", "round"), Ae(i, "stroke-width", "1.5"), Ae(i, "d", "M9 21h12"), Ae(n, "fill", "none"), Ae(t, "xmlns", "http://www.w3.org/2000/svg"), Ae(t, "width", "100%"), Ae(t, "height", "100%"), Ae(t, "viewBox", "0 0 24 24");
    },
    m(l, s) {
      ra(l, t, s), On(t, n), On(n, r), On(n, i);
    },
    p: Mn,
    i: Mn,
    o: Mn,
    d(l) {
      l && ta(t);
    }
  };
}
class sa extends ea {
  constructor(t) {
    super(), na(this, t, null, la, ia, {});
  }
}
const {
  SvelteComponent: oa,
  append: Rn,
  attr: Y,
  detach: aa,
  init: ua,
  insert: fa,
  noop: Dn,
  safe_not_equal: ca,
  svg_element: Kt
} = window.__gradio__svelte__internal;
function ha(e) {
  let t, n, r, i;
  return {
    c() {
      t = Kt("svg"), n = Kt("rect"), r = Kt("circle"), i = Kt("polyline"), Y(n, "x", "3"), Y(n, "y", "3"), Y(n, "width", "18"), Y(n, "height", "18"), Y(n, "rx", "2"), Y(n, "ry", "2"), Y(r, "cx", "8.5"), Y(r, "cy", "8.5"), Y(r, "r", "1.5"), Y(i, "points", "21 15 16 10 5 21"), Y(t, "xmlns", "http://www.w3.org/2000/svg"), Y(t, "width", "100%"), Y(t, "height", "100%"), Y(t, "viewBox", "0 0 24 24"), Y(t, "fill", "none"), Y(t, "stroke", "currentColor"), Y(t, "stroke-width", "1.5"), Y(t, "stroke-linecap", "round"), Y(t, "stroke-linejoin", "round"), Y(t, "class", "feather feather-image");
    },
    m(l, s) {
      fa(l, t, s), Rn(t, n), Rn(t, r), Rn(t, i);
    },
    p: Dn,
    i: Dn,
    o: Dn,
    d(l) {
      l && aa(t);
    }
  };
}
let wn = class extends oa {
  constructor(t) {
    super(), ua(this, t, null, ha, ca, {});
  }
};
const {
  SvelteComponent: _a,
  append: da,
  attr: st,
  detach: ma,
  init: ga,
  insert: pa,
  noop: Un,
  safe_not_equal: ba,
  svg_element: Wr
} = window.__gradio__svelte__internal;
function wa(e) {
  let t, n;
  return {
    c() {
      t = Wr("svg"), n = Wr("path"), st(n, "fill", "currentColor"), st(n, "d", "M13.75 2a2.25 2.25 0 0 1 2.236 2.002V4h1.764A2.25 2.25 0 0 1 20 6.25V11h-1.5V6.25a.75.75 0 0 0-.75-.75h-2.129c-.404.603-1.091 1-1.871 1h-3.5c-.78 0-1.467-.397-1.871-1H6.25a.75.75 0 0 0-.75.75v13.5c0 .414.336.75.75.75h4.78a3.99 3.99 0 0 0 .505 1.5H6.25A2.25 2.25 0 0 1 4 19.75V6.25A2.25 2.25 0 0 1 6.25 4h1.764a2.25 2.25 0 0 1 2.236-2h3.5Zm2.245 2.096L16 4.25c0-.052-.002-.103-.005-.154ZM13.75 3.5h-3.5a.75.75 0 0 0 0 1.5h3.5a.75.75 0 0 0 0-1.5ZM15 12a3 3 0 0 0-3 3v5c0 .556.151 1.077.415 1.524l3.494-3.494a2.25 2.25 0 0 1 3.182 0l3.494 3.494c.264-.447.415-.968.415-1.524v-5a3 3 0 0 0-3-3h-5Zm0 11a2.985 2.985 0 0 1-1.524-.415l3.494-3.494a.75.75 0 0 1 1.06 0l3.494 3.494A2.985 2.985 0 0 1 20 23h-5Zm5-7a1 1 0 1 1 0-2a1 1 0 0 1 0 2Z"), st(t, "xmlns", "http://www.w3.org/2000/svg"), st(t, "width", "100%"), st(t, "height", "100%"), st(t, "viewBox", "0 0 24 24");
    },
    m(r, i) {
      pa(r, t, i), da(t, n);
    },
    p: Un,
    i: Un,
    o: Un,
    d(r) {
      r && ma(t);
    }
  };
}
class va extends _a {
  constructor(t) {
    super(), ga(this, t, null, wa, ba, {});
  }
}
const {
  SvelteComponent: ya,
  append: Jr,
  attr: pe,
  detach: Ea,
  init: Sa,
  insert: ka,
  noop: Fn,
  safe_not_equal: Ta,
  svg_element: Gn
} = window.__gradio__svelte__internal;
function Ba(e) {
  let t, n, r;
  return {
    c() {
      t = Gn("svg"), n = Gn("polyline"), r = Gn("path"), pe(n, "points", "1 4 1 10 7 10"), pe(r, "d", "M3.51 15a9 9 0 1 0 2.13-9.36L1 10"), pe(t, "xmlns", "http://www.w3.org/2000/svg"), pe(t, "width", "100%"), pe(t, "height", "100%"), pe(t, "viewBox", "0 0 24 24"), pe(t, "fill", "none"), pe(t, "stroke", "currentColor"), pe(t, "stroke-width", "2"), pe(t, "stroke-linecap", "round"), pe(t, "stroke-linejoin", "round"), pe(t, "class", "feather feather-rotate-ccw");
    },
    m(i, l) {
      ka(i, t, l), Jr(t, n), Jr(t, r);
    },
    p: Fn,
    i: Fn,
    o: Fn,
    d(i) {
      i && Ea(t);
    }
  };
}
class Aa extends ya {
  constructor(t) {
    super(), Sa(this, t, null, Ba, Ta, {});
  }
}
const {
  SvelteComponent: Ca,
  append: qn,
  attr: re,
  detach: Ha,
  init: Pa,
  insert: Na,
  noop: xn,
  safe_not_equal: Ia,
  svg_element: $t
} = window.__gradio__svelte__internal;
function La(e) {
  let t, n, r, i;
  return {
    c() {
      t = $t("svg"), n = $t("path"), r = $t("polyline"), i = $t("line"), re(n, "d", "M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"), re(r, "points", "17 8 12 3 7 8"), re(i, "x1", "12"), re(i, "y1", "3"), re(i, "x2", "12"), re(i, "y2", "15"), re(t, "xmlns", "http://www.w3.org/2000/svg"), re(t, "width", "90%"), re(t, "height", "90%"), re(t, "viewBox", "0 0 24 24"), re(t, "fill", "none"), re(t, "stroke", "currentColor"), re(t, "stroke-width", "2"), re(t, "stroke-linecap", "round"), re(t, "stroke-linejoin", "round"), re(t, "class", "feather feather-upload");
    },
    m(l, s) {
      Na(l, t, s), qn(t, n), qn(t, r), qn(t, i);
    },
    p: xn,
    i: xn,
    o: xn,
    d(l) {
      l && Ha(t);
    }
  };
}
let Tl = class extends Ca {
  constructor(t) {
    super(), Pa(this, t, null, La, Ia, {});
  }
};
const {
  SvelteComponent: Oa,
  create_component: Ma,
  destroy_component: Ra,
  init: Da,
  mount_component: Ua,
  safe_not_equal: Fa,
  transition_in: Ga,
  transition_out: qa
} = window.__gradio__svelte__internal, { createEventDispatcher: xa } = window.__gradio__svelte__internal;
function ja(e) {
  let t, n;
  return t = new ht({
    props: {
      Icon: Vo,
      label: (
        /*i18n*/
        e[2]("common.share")
      ),
      pending: (
        /*pending*/
        e[3]
      )
    }
  }), t.$on(
    "click",
    /*click_handler*/
    e[5]
  ), {
    c() {
      Ma(t.$$.fragment);
    },
    m(r, i) {
      Ua(t, r, i), n = !0;
    },
    p(r, [i]) {
      const l = {};
      i & /*i18n*/
      4 && (l.label = /*i18n*/
      r[2]("common.share")), i & /*pending*/
      8 && (l.pending = /*pending*/
      r[3]), t.$set(l);
    },
    i(r) {
      n || (Ga(t.$$.fragment, r), n = !0);
    },
    o(r) {
      qa(t.$$.fragment, r), n = !1;
    },
    d(r) {
      Ra(t, r);
    }
  };
}
function za(e, t, n) {
  const r = xa();
  let { formatter: i } = t, { value: l } = t, { i18n: s } = t, a = !1;
  const o = async () => {
    try {
      n(3, a = !0);
      const u = await i(l);
      r("share", { description: u });
    } catch (u) {
      console.error(u);
      let f = u instanceof sn ? u.message : "Share failed.";
      r("error", f);
    } finally {
      n(3, a = !1);
    }
  };
  return e.$$set = (u) => {
    "formatter" in u && n(0, i = u.formatter), "value" in u && n(1, l = u.value), "i18n" in u && n(2, s = u.i18n);
  }, [i, l, s, a, r, o];
}
class Va extends Oa {
  constructor(t) {
    super(), Da(this, t, za, ja, Fa, { formatter: 0, value: 1, i18n: 2 });
  }
}
const {
  SvelteComponent: Xa,
  append: tt,
  attr: hr,
  create_component: Za,
  destroy_component: Wa,
  detach: on,
  element: _r,
  init: Ja,
  insert: an,
  mount_component: Ya,
  safe_not_equal: Qa,
  set_data: dr,
  space: mr,
  text: Ot,
  toggle_class: Yr,
  transition_in: Ka,
  transition_out: $a
} = window.__gradio__svelte__internal;
function Qr(e) {
  let t, n, r = (
    /*i18n*/
    e[1]("common.or") + ""
  ), i, l, s, a = (
    /*message*/
    (e[2] || /*i18n*/
    e[1]("upload_text.click_to_upload")) + ""
  ), o;
  return {
    c() {
      t = _r("span"), n = Ot("- "), i = Ot(r), l = Ot(" -"), s = mr(), o = Ot(a), hr(t, "class", "or svelte-kzcjhc");
    },
    m(u, f) {
      an(u, t, f), tt(t, n), tt(t, i), tt(t, l), an(u, s, f), an(u, o, f);
    },
    p(u, f) {
      f & /*i18n*/
      2 && r !== (r = /*i18n*/
      u[1]("common.or") + "") && dr(i, r), f & /*message, i18n*/
      6 && a !== (a = /*message*/
      (u[2] || /*i18n*/
      u[1]("upload_text.click_to_upload")) + "") && dr(o, a);
    },
    d(u) {
      u && (on(t), on(s), on(o));
    }
  };
}
function eu(e) {
  let t, n, r, i, l = (
    /*i18n*/
    e[1](
      /*defs*/
      e[5][
        /*type*/
        e[0]
      ] || /*defs*/
      e[5].file
    ) + ""
  ), s, a, o;
  r = new Tl({});
  let u = (
    /*mode*/
    e[3] !== "short" && Qr(e)
  );
  return {
    c() {
      t = _r("div"), n = _r("span"), Za(r.$$.fragment), i = mr(), s = Ot(l), a = mr(), u && u.c(), hr(n, "class", "icon-wrap svelte-kzcjhc"), Yr(
        n,
        "hovered",
        /*hovered*/
        e[4]
      ), hr(t, "class", "wrap svelte-kzcjhc");
    },
    m(f, c) {
      an(f, t, c), tt(t, n), Ya(r, n, null), tt(t, i), tt(t, s), tt(t, a), u && u.m(t, null), o = !0;
    },
    p(f, [c]) {
      (!o || c & /*hovered*/
      16) && Yr(
        n,
        "hovered",
        /*hovered*/
        f[4]
      ), (!o || c & /*i18n, type*/
      3) && l !== (l = /*i18n*/
      f[1](
        /*defs*/
        f[5][
          /*type*/
          f[0]
        ] || /*defs*/
        f[5].file
      ) + "") && dr(s, l), /*mode*/
      f[3] !== "short" ? u ? u.p(f, c) : (u = Qr(f), u.c(), u.m(t, null)) : u && (u.d(1), u = null);
    },
    i(f) {
      o || (Ka(r.$$.fragment, f), o = !0);
    },
    o(f) {
      $a(r.$$.fragment, f), o = !1;
    },
    d(f) {
      f && on(t), Wa(r), u && u.d();
    }
  };
}
function tu(e, t, n) {
  let { type: r = "file" } = t, { i18n: i } = t, { message: l = void 0 } = t, { mode: s = "full" } = t, { hovered: a = !1 } = t;
  const o = {
    image: "upload_text.drop_image",
    video: "upload_text.drop_video",
    audio: "upload_text.drop_audio",
    file: "upload_text.drop_file",
    csv: "upload_text.drop_csv"
  };
  return e.$$set = (u) => {
    "type" in u && n(0, r = u.type), "i18n" in u && n(1, i = u.i18n), "message" in u && n(2, l = u.message), "mode" in u && n(3, s = u.mode), "hovered" in u && n(4, a = u.hovered);
  }, [r, i, l, s, a, o];
}
class nu extends Xa {
  constructor(t) {
    super(), Ja(this, t, tu, eu, Qa, {
      type: 0,
      i18n: 1,
      message: 2,
      mode: 3,
      hovered: 4
    });
  }
}
const {
  SvelteComponent: ru,
  attr: iu,
  create_slot: lu,
  detach: su,
  element: ou,
  get_all_dirty_from_scope: au,
  get_slot_changes: uu,
  init: fu,
  insert: cu,
  safe_not_equal: hu,
  toggle_class: Kr,
  transition_in: _u,
  transition_out: du,
  update_slot_base: mu
} = window.__gradio__svelte__internal;
function gu(e) {
  let t, n;
  const r = (
    /*#slots*/
    e[2].default
  ), i = lu(
    r,
    e,
    /*$$scope*/
    e[1],
    null
  );
  return {
    c() {
      t = ou("div"), i && i.c(), iu(t, "class", "svelte-ipfyu7"), Kr(
        t,
        "show_border",
        /*show_border*/
        e[0]
      );
    },
    m(l, s) {
      cu(l, t, s), i && i.m(t, null), n = !0;
    },
    p(l, [s]) {
      i && i.p && (!n || s & /*$$scope*/
      2) && mu(
        i,
        r,
        l,
        /*$$scope*/
        l[1],
        n ? uu(
          r,
          /*$$scope*/
          l[1],
          s,
          null
        ) : au(
          /*$$scope*/
          l[1]
        ),
        null
      ), (!n || s & /*show_border*/
      1) && Kr(
        t,
        "show_border",
        /*show_border*/
        l[0]
      );
    },
    i(l) {
      n || (_u(i, l), n = !0);
    },
    o(l) {
      du(i, l), n = !1;
    },
    d(l) {
      l && su(t), i && i.d(l);
    }
  };
}
function pu(e, t, n) {
  let { $$slots: r = {}, $$scope: i } = t, { show_border: l = !1 } = t;
  return e.$$set = (s) => {
    "show_border" in s && n(0, l = s.show_border), "$$scope" in s && n(1, i = s.$$scope);
  }, [l, i, r];
}
class bu extends ru {
  constructor(t) {
    super(), fu(this, t, pu, gu, hu, { show_border: 0 });
  }
}
const Bl = (e) => {
  let t = e.currentTarget;
  const n = t.getBoundingClientRect(), r = t.naturalWidth / n.width, i = t.naturalHeight / n.height;
  if (r > i) {
    const a = t.naturalHeight / r, o = (n.height - a) / 2;
    var l = Math.round((e.clientX - n.left) * r), s = Math.round((e.clientY - n.top - o) * r);
  } else {
    const a = t.naturalWidth / i, o = (n.width - a) / 2;
    var l = Math.round((e.clientX - n.left - o) * i), s = Math.round((e.clientY - n.top) * i);
  }
  return l < 0 || l >= t.naturalWidth || s < 0 || s >= t.naturalHeight ? null : [l, s];
};
var jn = new Intl.Collator(0, { numeric: 1 }).compare;
function $r(e, t, n) {
  return e = e.split("."), t = t.split("."), jn(e[0], t[0]) || jn(e[1], t[1]) || (t[2] = t.slice(2).join("."), n = /[.-]/.test(e[2] = e.slice(2).join(".")), n == /[.-]/.test(t[2]) ? jn(e[2], t[2]) : n ? -1 : 1);
}
function Xe(e, t, n) {
  return t.startsWith("http://") || t.startsWith("https://") ? n ? e : t : e + t;
}
function zn(e) {
  if (e.startsWith("http")) {
    const { protocol: t, host: n } = new URL(e);
    return n.endsWith("hf.space") ? {
      ws_protocol: "wss",
      host: n,
      http_protocol: t
    } : {
      ws_protocol: t === "https:" ? "wss" : "ws",
      http_protocol: t,
      host: n
    };
  } else if (e.startsWith("file:"))
    return {
      ws_protocol: "ws",
      http_protocol: "http:",
      host: "lite.local"
      // Special fake hostname only used for this case. This matches the hostname allowed in `is_self_host()` in `js/wasm/network/host.ts`.
    };
  return {
    ws_protocol: "wss",
    http_protocol: "https:",
    host: e
  };
}
const Al = /^[^\/]*\/[^\/]*$/, wu = /.*hf\.space\/{0,1}$/;
async function vu(e, t) {
  const n = {};
  t && (n.Authorization = `Bearer ${t}`);
  const r = e.trim();
  if (Al.test(r))
    try {
      const i = await fetch(
        `https://huggingface.co/api/spaces/${r}/host`,
        { headers: n }
      );
      if (i.status !== 200)
        throw new Error("Space metadata could not be loaded.");
      const l = (await i.json()).host;
      return {
        space_id: e,
        ...zn(l)
      };
    } catch (i) {
      throw new Error("Space metadata could not be loaded." + i.message);
    }
  if (wu.test(r)) {
    const { ws_protocol: i, http_protocol: l, host: s } = zn(r);
    return {
      space_id: s.replace(".hf.space", ""),
      ws_protocol: i,
      http_protocol: l,
      host: s
    };
  }
  return {
    space_id: !1,
    ...zn(r)
  };
}
function yu(e) {
  let t = {};
  return e.forEach(({ api_name: n }, r) => {
    n && (t[n] = r);
  }), t;
}
const Eu = /^(?=[^]*\b[dD]iscussions{0,1}\b)(?=[^]*\b[dD]isabled\b)[^]*$/;
async function ei(e) {
  try {
    const n = (await fetch(
      `https://huggingface.co/api/spaces/${e}/discussions`,
      {
        method: "HEAD"
      }
    )).headers.get("x-error-message");
    return !(n && Eu.test(n));
  } catch {
    return !1;
  }
}
function Re(e, t, n) {
  if (e == null)
    return null;
  if (Array.isArray(e)) {
    const r = [];
    for (const i of e)
      i == null ? r.push(null) : r.push(Re(i, t, n));
    return r;
  }
  return e.is_stream ? n == null ? new _t({
    ...e,
    url: t + "/stream/" + e.path
  }) : new _t({
    ...e,
    url: "/proxy=" + n + "stream/" + e.path
  }) : new _t({
    ...e,
    url: ku(e.path, t, n)
  });
}
function Su(e) {
  try {
    const t = new URL(e);
    return t.protocol === "http:" || t.protocol === "https:";
  } catch {
    return !1;
  }
}
function ku(e, t, n) {
  return e == null ? n ? `/proxy=${n}file=` : `${t}/file=` : Su(e) ? e : n ? `/proxy=${n}file=${e}` : `${t}/file=${e}`;
}
async function Tu(e, t, n, r = Hu) {
  let i = (Array.isArray(e) ? e : [e]).map(
    (l) => l.blob
  );
  return await Promise.all(
    await r(t, i, void 0, n).then(
      async (l) => {
        if (l.error)
          throw new Error(l.error);
        return l.files ? l.files.map((s, a) => {
          const o = new _t({ ...e[a], path: s });
          return Re(o, t, null);
        }) : [];
      }
    )
  );
}
async function Bu(e, t) {
  return e.map(
    (n, r) => new _t({
      path: n.name,
      orig_name: n.name,
      blob: n,
      size: n.size,
      mime_type: n.type,
      is_stream: t
    })
  );
}
class _t {
  constructor({
    path: t,
    url: n,
    orig_name: r,
    size: i,
    blob: l,
    is_stream: s,
    mime_type: a,
    alt_text: o
  }) {
    this.path = t, this.url = n, this.orig_name = r, this.size = i, this.blob = n ? void 0 : l, this.is_stream = s, this.mime_type = a, this.alt_text = o;
  }
}
const Au = "This application is too busy. Keep trying!", It = "Connection errored out.";
let Cl;
function Cu(e, t) {
  return { post_data: n, upload_files: r, client: i, handle_blob: l };
  async function n(s, a, o) {
    const u = { "Content-Type": "application/json" };
    o && (u.Authorization = `Bearer ${o}`);
    try {
      var f = await e(s, {
        method: "POST",
        body: JSON.stringify(a),
        headers: u
      });
    } catch {
      return [{ error: It }, 500];
    }
    return [await f.json(), f.status];
  }
  async function r(s, a, o, u) {
    const f = {};
    o && (f.Authorization = `Bearer ${o}`);
    const c = 1e3, h = [];
    for (let d = 0; d < a.length; d += c) {
      const g = a.slice(d, d + c), S = new FormData();
      g.forEach((T) => {
        S.append("files", T);
      });
      try {
        const T = u ? `${s}/upload?upload_id=${u}` : `${s}/upload`;
        var _ = await e(T, {
          method: "POST",
          body: S,
          headers: f
        });
      } catch {
        return { error: It };
      }
      const b = await _.json();
      h.push(...b);
    }
    return { files: h };
  }
  async function i(s, a = { normalise_files: !0 }) {
    return new Promise(async (o) => {
      const { status_callback: u, hf_token: f, normalise_files: c } = a, h = {
        predict: M,
        submit: ie,
        view_api: $,
        component_server: oe
      }, _ = c ?? !0;
      if ((typeof window > "u" || !("WebSocket" in window)) && !global.Websocket) {
        const H = await import("./wrapper-6f348d45-f837cf34.js");
        Cl = (await import("./__vite-browser-external-2447137e.js")).Blob, global.WebSocket = H.WebSocket;
      }
      const { ws_protocol: d, http_protocol: g, host: S, space_id: b } = await vu(s, f), T = Math.random().toString(36).substring(2), y = {};
      let m, v = {}, G = !1;
      f && b && (G = await Nu(b, f));
      async function F(H) {
        if (m = H, v = yu((H == null ? void 0 : H.dependencies) || []), m.auth_required)
          return {
            config: m,
            ...h
          };
        try {
          q = await $(m);
        } catch (I) {
          console.error(`Could not get api details: ${I.message}`);
        }
        return {
          config: m,
          ...h
        };
      }
      let q;
      async function K(H) {
        if (u && u(H), H.status === "running")
          try {
            m = await ii(
              e,
              `${g}//${S}`,
              f
            );
            const I = await F(m);
            o(I);
          } catch (I) {
            console.error(I), u && u({
              status: "error",
              message: "Could not load this space.",
              load_status: "error",
              detail: "NOT_FOUND"
            });
          }
      }
      try {
        m = await ii(
          e,
          `${g}//${S}`,
          f
        );
        const H = await F(m);
        o(H);
      } catch (H) {
        console.error(H), b ? pr(
          b,
          Al.test(b) ? "space_name" : "subdomain",
          K
        ) : u && u({
          status: "error",
          message: "Could not load this space.",
          load_status: "error",
          detail: "NOT_FOUND"
        });
      }
      function M(H, I, X) {
        let x = !1, C = !1, B;
        if (typeof H == "number")
          B = m.dependencies[H];
        else {
          const w = H.replace(/^\//, "");
          B = m.dependencies[v[w]];
        }
        if (B.types.continuous)
          throw new Error(
            "Cannot call predict on this function as it may run forever. Use submit instead"
          );
        return new Promise((w, E) => {
          const A = ie(H, I, X);
          let p;
          A.on("data", (D) => {
            C && (A.destroy(), w(D)), x = !0, p = D;
          }).on("status", (D) => {
            D.stage === "error" && E(D), D.stage === "complete" && (C = !0, x && (A.destroy(), w(p)));
          });
        });
      }
      function ie(H, I, X, x = null) {
        let C, B;
        if (typeof H == "number")
          C = H, B = q.unnamed_endpoints[C];
        else {
          const ee = H.replace(/^\//, "");
          C = v[ee], B = q.named_endpoints[H.trim()];
        }
        if (typeof C != "number")
          throw new Error(
            "There is no endpoint matching that name of fn_index matching that number."
          );
        let w, E, A = m.protocol ?? "sse";
        const p = typeof H == "number" ? "/predict" : H;
        let D, Se = null, le = !1;
        const ke = {};
        let _e = "";
        typeof window < "u" && (_e = new URLSearchParams(window.location.search).toString()), l(
          `${g}//${Xe(S, m.path, !0)}`,
          I,
          B,
          f
        ).then((ee) => {
          if (D = { data: ee || [], event_data: X, fn_index: C, trigger_id: x }, Iu(C, m))
            j({
              type: "status",
              endpoint: p,
              stage: "pending",
              queue: !1,
              fn_index: C,
              time: /* @__PURE__ */ new Date()
            }), n(
              `${g}//${Xe(S, m.path, !0)}/run${p.startsWith("/") ? p : `/${p}`}${_e ? "?" + _e : ""}`,
              {
                ...D,
                session_hash: T
              },
              f
            ).then(([W, te]) => {
              const qe = _ ? Vn(
                W.data,
                B,
                m.root,
                m.root_url
              ) : W.data;
              te == 200 ? (j({
                type: "data",
                endpoint: p,
                fn_index: C,
                data: qe,
                time: /* @__PURE__ */ new Date()
              }), j({
                type: "status",
                endpoint: p,
                fn_index: C,
                stage: "complete",
                eta: W.average_duration,
                queue: !1,
                time: /* @__PURE__ */ new Date()
              })) : j({
                type: "status",
                stage: "error",
                endpoint: p,
                fn_index: C,
                message: W.error,
                queue: !1,
                time: /* @__PURE__ */ new Date()
              });
            }).catch((W) => {
              j({
                type: "status",
                stage: "error",
                message: W.message,
                endpoint: p,
                fn_index: C,
                queue: !1,
                time: /* @__PURE__ */ new Date()
              });
            });
          else if (A == "ws") {
            j({
              type: "status",
              stage: "pending",
              queue: !0,
              endpoint: p,
              fn_index: C,
              time: /* @__PURE__ */ new Date()
            });
            let W = new URL(`${d}://${Xe(
              S,
              m.path,
              !0
            )}
							/queue/join${_e ? "?" + _e : ""}`);
            G && W.searchParams.set("__sign", G), w = t(W), w.onclose = (te) => {
              te.wasClean || j({
                type: "status",
                stage: "error",
                broken: !0,
                message: It,
                queue: !0,
                endpoint: p,
                fn_index: C,
                time: /* @__PURE__ */ new Date()
              });
            }, w.onmessage = function(te) {
              const qe = JSON.parse(te.data), { type: me, status: Q, data: Te } = li(
                qe,
                y[C]
              );
              if (me === "update" && Q && !le)
                j({
                  type: "status",
                  endpoint: p,
                  fn_index: C,
                  time: /* @__PURE__ */ new Date(),
                  ...Q
                }), Q.stage === "error" && w.close();
              else if (me === "hash") {
                w.send(JSON.stringify({ fn_index: C, session_hash: T }));
                return;
              } else
                me === "data" ? w.send(JSON.stringify({ ...D, session_hash: T })) : me === "complete" ? le = Q : me === "log" ? j({
                  type: "log",
                  log: Te.log,
                  level: Te.level,
                  endpoint: p,
                  fn_index: C
                }) : me === "generating" && j({
                  type: "status",
                  time: /* @__PURE__ */ new Date(),
                  ...Q,
                  stage: Q == null ? void 0 : Q.stage,
                  queue: !0,
                  endpoint: p,
                  fn_index: C
                });
              Te && (j({
                type: "data",
                time: /* @__PURE__ */ new Date(),
                data: _ ? Vn(
                  Te.data,
                  B,
                  m.root,
                  m.root_url
                ) : Te.data,
                endpoint: p,
                fn_index: C
              }), le && (j({
                type: "status",
                time: /* @__PURE__ */ new Date(),
                ...le,
                stage: Q == null ? void 0 : Q.stage,
                queue: !0,
                endpoint: p,
                fn_index: C
              }), w.close()));
            }, $r(m.version || "2.0.0", "3.6") < 0 && addEventListener(
              "open",
              () => w.send(JSON.stringify({ hash: T }))
            );
          } else {
            j({
              type: "status",
              stage: "pending",
              queue: !0,
              endpoint: p,
              fn_index: C,
              time: /* @__PURE__ */ new Date()
            });
            var de = new URLSearchParams({
              fn_index: C.toString(),
              session_hash: T
            }).toString();
            let W = new URL(
              `${g}//${Xe(
                S,
                m.path,
                !0
              )}/queue/join?${_e ? _e + "&" : ""}${de}`
            );
            E = new EventSource(W), E.onmessage = async function(te) {
              const qe = JSON.parse(te.data), { type: me, status: Q, data: Te } = li(
                qe,
                y[C]
              );
              if (me === "update" && Q && !le)
                j({
                  type: "status",
                  endpoint: p,
                  fn_index: C,
                  time: /* @__PURE__ */ new Date(),
                  ...Q
                }), Q.stage === "error" && E.close();
              else if (me === "data") {
                Se = qe.event_id;
                let [z_, bs] = await n(
                  `${g}//${Xe(
                    S,
                    m.path,
                    !0
                  )}/queue/data`,
                  {
                    ...D,
                    session_hash: T,
                    event_id: Se
                  },
                  f
                );
                bs !== 200 && (j({
                  type: "status",
                  stage: "error",
                  message: It,
                  queue: !0,
                  endpoint: p,
                  fn_index: C,
                  time: /* @__PURE__ */ new Date()
                }), E.close());
              } else
                me === "complete" ? le = Q : me === "log" ? j({
                  type: "log",
                  log: Te.log,
                  level: Te.level,
                  endpoint: p,
                  fn_index: C
                }) : me === "generating" && j({
                  type: "status",
                  time: /* @__PURE__ */ new Date(),
                  ...Q,
                  stage: Q == null ? void 0 : Q.stage,
                  queue: !0,
                  endpoint: p,
                  fn_index: C
                });
              Te && (j({
                type: "data",
                time: /* @__PURE__ */ new Date(),
                data: _ ? Vn(
                  Te.data,
                  B,
                  m.root,
                  m.root_url
                ) : Te.data,
                endpoint: p,
                fn_index: C
              }), le && (j({
                type: "status",
                time: /* @__PURE__ */ new Date(),
                ...le,
                stage: Q == null ? void 0 : Q.stage,
                queue: !0,
                endpoint: p,
                fn_index: C
              }), E.close()));
            };
          }
        });
        function j(ee) {
          const W = ke[ee.type] || [];
          W == null || W.forEach((te) => te(ee));
        }
        function k(ee, de) {
          const W = ke, te = W[ee] || [];
          return W[ee] = te, te == null || te.push(de), { on: k, off: Zt, cancel: Bn, destroy: An };
        }
        function Zt(ee, de) {
          const W = ke;
          let te = W[ee] || [];
          return te = te == null ? void 0 : te.filter((qe) => qe !== de), W[ee] = te, { on: k, off: Zt, cancel: Bn, destroy: An };
        }
        async function Bn() {
          const ee = {
            stage: "complete",
            queue: !1,
            time: /* @__PURE__ */ new Date()
          };
          le = ee, j({
            ...ee,
            type: "status",
            endpoint: p,
            fn_index: C
          });
          let de = {};
          A === "ws" ? (w && w.readyState === 0 ? w.addEventListener("open", () => {
            w.close();
          }) : w.close(), de = { fn_index: C, session_hash: T }) : (E.close(), de = { event_id: Se });
          try {
            await e(
              `${g}//${Xe(
                S,
                m.path,
                !0
              )}/reset`,
              {
                headers: { "Content-Type": "application/json" },
                method: "POST",
                body: JSON.stringify(de)
              }
            );
          } catch {
            console.warn(
              "The `/reset` endpoint could not be called. Subsequent endpoint results may be unreliable."
            );
          }
        }
        function An() {
          for (const ee in ke)
            ke[ee].forEach((de) => {
              Zt(ee, de);
            });
        }
        return {
          on: k,
          off: Zt,
          cancel: Bn,
          destroy: An
        };
      }
      async function oe(H, I, X) {
        var x;
        const C = { "Content-Type": "application/json" };
        f && (C.Authorization = `Bearer ${f}`);
        let B, w = m.components.find(
          (p) => p.id === H
        );
        (x = w == null ? void 0 : w.props) != null && x.root_url ? B = w.props.root_url : B = `${g}//${Xe(
          S,
          m.path,
          !0
        )}/`;
        const E = await e(
          `${B}component_server/`,
          {
            method: "POST",
            body: JSON.stringify({
              data: X,
              component_id: H,
              fn_name: I,
              session_hash: T
            }),
            headers: C
          }
        );
        if (!E.ok)
          throw new Error(
            "Could not connect to component server: " + E.statusText
          );
        return await E.json();
      }
      async function $(H) {
        if (q)
          return q;
        const I = { "Content-Type": "application/json" };
        f && (I.Authorization = `Bearer ${f}`);
        let X;
        if ($r(H.version || "2.0.0", "3.30") < 0 ? X = await e(
          "https://gradio-space-api-fetcher-v2.hf.space/api",
          {
            method: "POST",
            body: JSON.stringify({
              serialize: !1,
              config: JSON.stringify(H)
            }),
            headers: I
          }
        ) : X = await e(`${H.root}/info`, {
          headers: I
        }), !X.ok)
          throw new Error(It);
        let x = await X.json();
        return "api" in x && (x = x.api), x.named_endpoints["/predict"] && !x.unnamed_endpoints[0] && (x.unnamed_endpoints[0] = x.named_endpoints["/predict"]), Pu(x, H, v);
      }
    });
  }
  async function l(s, a, o, u) {
    const f = await gr(
      a,
      void 0,
      [],
      !0,
      o
    );
    return Promise.all(
      f.map(async ({ path: c, blob: h, type: _ }) => {
        if (h) {
          const d = (await r(s, [h], u)).files[0];
          return { path: c, file_url: d, type: _, name: h == null ? void 0 : h.name };
        }
        return { path: c, type: _ };
      })
    ).then((c) => (c.forEach(({ path: h, file_url: _, type: d, name: g }) => {
      if (d === "Gallery")
        ri(a, _, h);
      else if (_) {
        const S = new _t({ path: _, orig_name: g });
        ri(a, S, h);
      }
    }), a));
  }
}
const { post_data: Z_, upload_files: Hu, client: W_, handle_blob: J_ } = Cu(
  fetch,
  (...e) => new WebSocket(...e)
);
function Vn(e, t, n, r) {
  return e.map((i, l) => {
    var s, a, o, u;
    return ((a = (s = t == null ? void 0 : t.returns) == null ? void 0 : s[l]) == null ? void 0 : a.component) === "File" ? Re(i, n, r) : ((u = (o = t == null ? void 0 : t.returns) == null ? void 0 : o[l]) == null ? void 0 : u.component) === "Gallery" ? i.map((f) => Array.isArray(f) ? [Re(f[0], n, r), f[1]] : [Re(f, n, r), null]) : typeof i == "object" && i.path ? Re(i, n, r) : i;
  });
}
function ti(e, t, n, r) {
  switch (e.type) {
    case "string":
      return "string";
    case "boolean":
      return "boolean";
    case "number":
      return "number";
  }
  if (n === "JSONSerializable" || n === "StringSerializable")
    return "any";
  if (n === "ListStringSerializable")
    return "string[]";
  if (t === "Image")
    return r === "parameter" ? "Blob | File | Buffer" : "string";
  if (n === "FileSerializable")
    return (e == null ? void 0 : e.type) === "array" ? r === "parameter" ? "(Blob | File | Buffer)[]" : "{ name: string; data: string; size?: number; is_file?: boolean; orig_name?: string}[]" : r === "parameter" ? "Blob | File | Buffer" : "{ name: string; data: string; size?: number; is_file?: boolean; orig_name?: string}";
  if (n === "GallerySerializable")
    return r === "parameter" ? "[(Blob | File | Buffer), (string | null)][]" : "[{ name: string; data: string; size?: number; is_file?: boolean; orig_name?: string}, (string | null))][]";
}
function ni(e, t) {
  return t === "GallerySerializable" ? "array of [file, label] tuples" : t === "ListStringSerializable" ? "array of strings" : t === "FileSerializable" ? "array of files or single file" : e.description;
}
function Pu(e, t, n) {
  const r = {
    named_endpoints: {},
    unnamed_endpoints: {}
  };
  for (const i in e) {
    const l = e[i];
    for (const s in l) {
      const a = t.dependencies[s] ? s : n[s.replace("/", "")], o = l[s];
      r[i][s] = {}, r[i][s].parameters = {}, r[i][s].returns = {}, r[i][s].type = t.dependencies[a].types, r[i][s].parameters = o.parameters.map(
        ({ label: u, component: f, type: c, serializer: h }) => ({
          label: u,
          component: f,
          type: ti(c, f, h, "parameter"),
          description: ni(c, h)
        })
      ), r[i][s].returns = o.returns.map(
        ({ label: u, component: f, type: c, serializer: h }) => ({
          label: u,
          component: f,
          type: ti(c, f, h, "return"),
          description: ni(c, h)
        })
      );
    }
  }
  return r;
}
async function Nu(e, t) {
  try {
    return (await (await fetch(`https://huggingface.co/api/spaces/${e}/jwt`, {
      headers: {
        Authorization: `Bearer ${t}`
      }
    })).json()).token || !1;
  } catch (n) {
    return console.error(n), !1;
  }
}
function ri(e, t, n) {
  for (; n.length > 1; )
    e = e[n.shift()];
  e[n.shift()] = t;
}
async function gr(e, t = void 0, n = [], r = !1, i = void 0) {
  if (Array.isArray(e)) {
    let l = [];
    return await Promise.all(
      e.map(async (s, a) => {
        var o;
        let u = n.slice();
        u.push(a);
        const f = await gr(
          e[a],
          r ? ((o = i == null ? void 0 : i.parameters[a]) == null ? void 0 : o.component) || void 0 : t,
          u,
          !1,
          i
        );
        l = l.concat(f);
      })
    ), l;
  } else {
    if (globalThis.Buffer && e instanceof globalThis.Buffer)
      return [
        {
          path: n,
          blob: t === "Image" ? !1 : new Cl([e]),
          type: t
        }
      ];
    if (typeof e == "object") {
      let l = [];
      for (let s in e)
        if (e.hasOwnProperty(s)) {
          let a = n.slice();
          a.push(s), l = l.concat(
            await gr(
              e[s],
              void 0,
              a,
              !1,
              i
            )
          );
        }
      return l;
    }
  }
  return [];
}
function Iu(e, t) {
  var n, r, i, l;
  return !(((r = (n = t == null ? void 0 : t.dependencies) == null ? void 0 : n[e]) == null ? void 0 : r.queue) === null ? t.enable_queue : (l = (i = t == null ? void 0 : t.dependencies) == null ? void 0 : i[e]) != null && l.queue) || !1;
}
async function ii(e, t, n) {
  const r = {};
  if (n && (r.Authorization = `Bearer ${n}`), typeof window < "u" && window.gradio_config && location.origin !== "http://localhost:9876" && !window.gradio_config.dev_mode) {
    const i = window.gradio_config.root, l = window.gradio_config;
    return l.root = Xe(t, l.root, !1), { ...l, path: i };
  } else if (t) {
    let i = await e(`${t}/config`, {
      headers: r
    });
    if (i.status === 200) {
      const l = await i.json();
      return l.path = l.path ?? "", l.root = t, l;
    }
    throw new Error("Could not get config.");
  }
  throw new Error("No config or app endpoint found");
}
async function pr(e, t, n) {
  let r = t === "subdomain" ? `https://huggingface.co/api/spaces/by-subdomain/${e}` : `https://huggingface.co/api/spaces/${e}`, i, l;
  try {
    if (i = await fetch(r), l = i.status, l !== 200)
      throw new Error();
    i = await i.json();
  } catch {
    n({
      status: "error",
      load_status: "error",
      message: "Could not get space status",
      detail: "NOT_FOUND"
    });
    return;
  }
  if (!i || l !== 200)
    return;
  const {
    runtime: { stage: s },
    id: a
  } = i;
  switch (s) {
    case "STOPPED":
    case "SLEEPING":
      n({
        status: "sleeping",
        load_status: "pending",
        message: "Space is asleep. Waking it up...",
        detail: s
      }), setTimeout(() => {
        pr(e, t, n);
      }, 1e3);
      break;
    case "PAUSED":
      n({
        status: "paused",
        load_status: "error",
        message: "This space has been paused by the author. If you would like to try this demo, consider duplicating the space.",
        detail: s,
        discussions_enabled: await ei(a)
      });
      break;
    case "RUNNING":
    case "RUNNING_BUILDING":
      n({
        status: "running",
        load_status: "complete",
        message: "",
        detail: s
      });
      break;
    case "BUILDING":
      n({
        status: "building",
        load_status: "pending",
        message: "Space is building...",
        detail: s
      }), setTimeout(() => {
        pr(e, t, n);
      }, 1e3);
      break;
    default:
      n({
        status: "space_error",
        load_status: "error",
        message: "This space is experiencing an issue.",
        detail: s,
        discussions_enabled: await ei(a)
      });
      break;
  }
}
function li(e, t) {
  switch (e.msg) {
    case "send_data":
      return { type: "data" };
    case "send_hash":
      return { type: "hash" };
    case "queue_full":
      return {
        type: "update",
        status: {
          queue: !0,
          message: Au,
          stage: "error",
          code: e.code,
          success: e.success
        }
      };
    case "estimation":
      return {
        type: "update",
        status: {
          queue: !0,
          stage: t || "pending",
          code: e.code,
          size: e.queue_size,
          position: e.rank,
          eta: e.rank_eta,
          success: e.success
        }
      };
    case "progress":
      return {
        type: "update",
        status: {
          queue: !0,
          stage: "pending",
          code: e.code,
          progress_data: e.progress_data,
          success: e.success
        }
      };
    case "log":
      return { type: "log", data: e };
    case "process_generating":
      return {
        type: "generating",
        status: {
          queue: !0,
          message: e.success ? null : e.output.error,
          stage: e.success ? "generating" : "error",
          code: e.code,
          progress_data: e.progress_data,
          eta: e.average_duration
        },
        data: e.success ? e.output : null
      };
    case "process_completed":
      return "error" in e.output ? {
        type: "update",
        status: {
          queue: !0,
          message: e.output.error,
          stage: "error",
          code: e.code,
          success: e.success
        }
      } : {
        type: "complete",
        status: {
          queue: !0,
          message: e.success ? void 0 : e.output.error,
          stage: e.success ? "complete" : "error",
          code: e.code,
          progress_data: e.progress_data,
          eta: e.output.average_duration
        },
        data: e.success ? e.output : null
      };
    case "process_starts":
      return {
        type: "update",
        status: {
          queue: !0,
          stage: "pending",
          code: e.code,
          size: e.rank,
          position: 0,
          success: e.success
        }
      };
  }
  return { type: "none", status: { stage: "error", queue: !0 } };
}
const {
  SvelteComponent: Lu,
  append: si,
  attr: ye,
  bubble: oi,
  check_outros: br,
  create_component: qt,
  destroy_component: xt,
  detach: dt,
  element: un,
  empty: Ou,
  group_outros: wr,
  init: Mu,
  insert: mt,
  listen: Ru,
  mount_component: jt,
  safe_not_equal: Du,
  space: vr,
  src_url_equal: ai,
  toggle_class: ui,
  transition_in: fe,
  transition_out: Pe
} = window.__gradio__svelte__internal, { createEventDispatcher: Uu } = window.__gradio__svelte__internal;
function Fu(e) {
  let t, n, r, i, l, s, a, o, u, f = (
    /*show_download_button*/
    e[3] && fi(e)
  ), c = (
    /*show_share_button*/
    e[5] && ci(e)
  );
  return {
    c() {
      t = un("div"), f && f.c(), n = vr(), c && c.c(), r = vr(), i = un("button"), l = un("img"), ye(t, "class", "icon-buttons svelte-1e0ed51"), ai(l.src, s = /*value*/
      e[0].url) || ye(l, "src", s), ye(l, "alt", ""), ye(l, "loading", "lazy"), ye(l, "class", "svelte-1e0ed51"), ui(
        l,
        "selectable",
        /*selectable*/
        e[4]
      ), ye(i, "class", "svelte-1e0ed51");
    },
    m(h, _) {
      mt(h, t, _), f && f.m(t, null), si(t, n), c && c.m(t, null), mt(h, r, _), mt(h, i, _), si(i, l), a = !0, o || (u = Ru(
        i,
        "click",
        /*handle_click*/
        e[7]
      ), o = !0);
    },
    p(h, _) {
      /*show_download_button*/
      h[3] ? f ? (f.p(h, _), _ & /*show_download_button*/
      8 && fe(f, 1)) : (f = fi(h), f.c(), fe(f, 1), f.m(t, n)) : f && (wr(), Pe(f, 1, 1, () => {
        f = null;
      }), br()), /*show_share_button*/
      h[5] ? c ? (c.p(h, _), _ & /*show_share_button*/
      32 && fe(c, 1)) : (c = ci(h), c.c(), fe(c, 1), c.m(t, null)) : c && (wr(), Pe(c, 1, 1, () => {
        c = null;
      }), br()), (!a || _ & /*value*/
      1 && !ai(l.src, s = /*value*/
      h[0].url)) && ye(l, "src", s), (!a || _ & /*selectable*/
      16) && ui(
        l,
        "selectable",
        /*selectable*/
        h[4]
      );
    },
    i(h) {
      a || (fe(f), fe(c), a = !0);
    },
    o(h) {
      Pe(f), Pe(c), a = !1;
    },
    d(h) {
      h && (dt(t), dt(r), dt(i)), f && f.d(), c && c.d(), o = !1, u();
    }
  };
}
function Gu(e) {
  let t, n;
  return t = new kl({
    props: {
      unpadded_box: !0,
      size: "large",
      $$slots: { default: [qu] },
      $$scope: { ctx: e }
    }
  }), {
    c() {
      qt(t.$$.fragment);
    },
    m(r, i) {
      jt(t, r, i), n = !0;
    },
    p(r, i) {
      const l = {};
      i & /*$$scope*/
      4096 && (l.$$scope = { dirty: i, ctx: r }), t.$set(l);
    },
    i(r) {
      n || (fe(t.$$.fragment, r), n = !0);
    },
    o(r) {
      Pe(t.$$.fragment, r), n = !1;
    },
    d(r) {
      xt(t, r);
    }
  };
}
function fi(e) {
  let t, n, r, i, l;
  return n = new ht({
    props: {
      Icon: $o,
      label: (
        /*i18n*/
        e[6]("common.download")
      )
    }
  }), {
    c() {
      t = un("a"), qt(n.$$.fragment), ye(t, "href", r = /*value*/
      e[0].url), ye(t, "target", window.__is_colab__ ? "_blank" : null), ye(t, "download", i = /*value*/
      e[0].orig_name || "image");
    },
    m(s, a) {
      mt(s, t, a), jt(n, t, null), l = !0;
    },
    p(s, a) {
      const o = {};
      a & /*i18n*/
      64 && (o.label = /*i18n*/
      s[6]("common.download")), n.$set(o), (!l || a & /*value*/
      1 && r !== (r = /*value*/
      s[0].url)) && ye(t, "href", r), (!l || a & /*value*/
      1 && i !== (i = /*value*/
      s[0].orig_name || "image")) && ye(t, "download", i);
    },
    i(s) {
      l || (fe(n.$$.fragment, s), l = !0);
    },
    o(s) {
      Pe(n.$$.fragment, s), l = !1;
    },
    d(s) {
      s && dt(t), xt(n);
    }
  };
}
function ci(e) {
  let t, n;
  return t = new Va({
    props: {
      i18n: (
        /*i18n*/
        e[6]
      ),
      formatter: (
        /*func*/
        e[8]
      ),
      value: (
        /*value*/
        e[0]
      )
    }
  }), t.$on(
    "share",
    /*share_handler*/
    e[9]
  ), t.$on(
    "error",
    /*error_handler*/
    e[10]
  ), {
    c() {
      qt(t.$$.fragment);
    },
    m(r, i) {
      jt(t, r, i), n = !0;
    },
    p(r, i) {
      const l = {};
      i & /*i18n*/
      64 && (l.i18n = /*i18n*/
      r[6]), i & /*value*/
      1 && (l.value = /*value*/
      r[0]), t.$set(l);
    },
    i(r) {
      n || (fe(t.$$.fragment, r), n = !0);
    },
    o(r) {
      Pe(t.$$.fragment, r), n = !1;
    },
    d(r) {
      xt(t, r);
    }
  };
}
function qu(e) {
  let t, n;
  return t = new wn({}), {
    c() {
      qt(t.$$.fragment);
    },
    m(r, i) {
      jt(t, r, i), n = !0;
    },
    i(r) {
      n || (fe(t.$$.fragment, r), n = !0);
    },
    o(r) {
      Pe(t.$$.fragment, r), n = !1;
    },
    d(r) {
      xt(t, r);
    }
  };
}
function xu(e) {
  let t, n, r, i, l, s;
  t = new yl({
    props: {
      show_label: (
        /*show_label*/
        e[2]
      ),
      Icon: wn,
      label: (
        /*label*/
        e[1] || /*i18n*/
        e[6]("image.image")
      )
    }
  });
  const a = [Gu, Fu], o = [];
  function u(f, c) {
    return (
      /*value*/
      f[0] === null || !/*value*/
      f[0].url ? 0 : 1
    );
  }
  return r = u(e), i = o[r] = a[r](e), {
    c() {
      qt(t.$$.fragment), n = vr(), i.c(), l = Ou();
    },
    m(f, c) {
      jt(t, f, c), mt(f, n, c), o[r].m(f, c), mt(f, l, c), s = !0;
    },
    p(f, [c]) {
      const h = {};
      c & /*show_label*/
      4 && (h.show_label = /*show_label*/
      f[2]), c & /*label, i18n*/
      66 && (h.label = /*label*/
      f[1] || /*i18n*/
      f[6]("image.image")), t.$set(h);
      let _ = r;
      r = u(f), r === _ ? o[r].p(f, c) : (wr(), Pe(o[_], 1, 1, () => {
        o[_] = null;
      }), br(), i = o[r], i ? i.p(f, c) : (i = o[r] = a[r](f), i.c()), fe(i, 1), i.m(l.parentNode, l));
    },
    i(f) {
      s || (fe(t.$$.fragment, f), fe(i), s = !0);
    },
    o(f) {
      Pe(t.$$.fragment, f), Pe(i), s = !1;
    },
    d(f) {
      f && (dt(n), dt(l)), xt(t, f), o[r].d(f);
    }
  };
}
function ju(e, t, n) {
  let { value: r } = t, { label: i = void 0 } = t, { show_label: l } = t, { show_download_button: s = !0 } = t, { selectable: a = !1 } = t, { show_share_button: o = !1 } = t, { i18n: u } = t;
  const f = Uu(), c = (g) => {
    let S = Bl(g);
    S && f("select", { index: S, value: null });
  }, h = async (g) => g ? `<img src="${await ys(g, "base64")}" />` : "";
  function _(g) {
    oi.call(this, e, g);
  }
  function d(g) {
    oi.call(this, e, g);
  }
  return e.$$set = (g) => {
    "value" in g && n(0, r = g.value), "label" in g && n(1, i = g.label), "show_label" in g && n(2, l = g.show_label), "show_download_button" in g && n(3, s = g.show_download_button), "selectable" in g && n(4, a = g.selectable), "show_share_button" in g && n(5, o = g.show_share_button), "i18n" in g && n(6, u = g.i18n);
  }, [
    r,
    i,
    l,
    s,
    a,
    o,
    u,
    c,
    h,
    _,
    d
  ];
}
class zu extends Lu {
  constructor(t) {
    super(), Mu(this, t, ju, xu, Du, {
      value: 0,
      label: 1,
      show_label: 2,
      show_download_button: 3,
      selectable: 4,
      show_share_button: 5,
      i18n: 6
    });
  }
}
function rt() {
}
function Vu(e) {
  return e();
}
function Xu(e) {
  e.forEach(Vu);
}
function Zu(e) {
  return typeof e == "function";
}
function Wu(e, t) {
  return e != e ? t == t : e !== t || e && typeof e == "object" || typeof e == "function";
}
function Ju(e, ...t) {
  if (e == null) {
    for (const r of t)
      r(void 0);
    return rt;
  }
  const n = e.subscribe(...t);
  return n.unsubscribe ? () => n.unsubscribe() : n;
}
const Hl = typeof window < "u";
let hi = Hl ? () => window.performance.now() : () => Date.now(), Pl = Hl ? (e) => requestAnimationFrame(e) : rt;
const gt = /* @__PURE__ */ new Set();
function Nl(e) {
  gt.forEach((t) => {
    t.c(e) || (gt.delete(t), t.f());
  }), gt.size !== 0 && Pl(Nl);
}
function Yu(e) {
  let t;
  return gt.size === 0 && Pl(Nl), {
    promise: new Promise((n) => {
      gt.add(t = { c: e, f: n });
    }),
    abort() {
      gt.delete(t);
    }
  };
}
const ot = [];
function Qu(e, t) {
  return {
    subscribe: zt(e, t).subscribe
  };
}
function zt(e, t = rt) {
  let n;
  const r = /* @__PURE__ */ new Set();
  function i(a) {
    if (Wu(e, a) && (e = a, n)) {
      const o = !ot.length;
      for (const u of r)
        u[1](), ot.push(u, e);
      if (o) {
        for (let u = 0; u < ot.length; u += 2)
          ot[u][0](ot[u + 1]);
        ot.length = 0;
      }
    }
  }
  function l(a) {
    i(a(e));
  }
  function s(a, o = rt) {
    const u = [a, o];
    return r.add(u), r.size === 1 && (n = t(i, l) || rt), a(e), () => {
      r.delete(u), r.size === 0 && n && (n(), n = null);
    };
  }
  return { set: i, update: l, subscribe: s };
}
function Ct(e, t, n) {
  const r = !Array.isArray(e), i = r ? [e] : e;
  if (!i.every(Boolean))
    throw new Error("derived() expects stores as input, got a falsy value");
  const l = t.length < 2;
  return Qu(n, (s, a) => {
    let o = !1;
    const u = [];
    let f = 0, c = rt;
    const h = () => {
      if (f)
        return;
      c();
      const d = t(r ? u[0] : u, s, a);
      l ? s(d) : c = Zu(d) ? d : rt;
    }, _ = i.map(
      (d, g) => Ju(
        d,
        (S) => {
          u[g] = S, f &= ~(1 << g), o && h();
        },
        () => {
          f |= 1 << g;
        }
      )
    );
    return o = !0, h(), function() {
      Xu(_), c(), o = !1;
    };
  });
}
function _i(e) {
  return Object.prototype.toString.call(e) === "[object Date]";
}
function yr(e, t, n, r) {
  if (typeof n == "number" || _i(n)) {
    const i = r - n, l = (n - t) / (e.dt || 1 / 60), s = e.opts.stiffness * i, a = e.opts.damping * l, o = (s - a) * e.inv_mass, u = (l + o) * e.dt;
    return Math.abs(u) < e.opts.precision && Math.abs(i) < e.opts.precision ? r : (e.settled = !1, _i(n) ? new Date(n.getTime() + u) : n + u);
  } else {
    if (Array.isArray(n))
      return n.map(
        (i, l) => yr(e, t[l], n[l], r[l])
      );
    if (typeof n == "object") {
      const i = {};
      for (const l in n)
        i[l] = yr(e, t[l], n[l], r[l]);
      return i;
    } else
      throw new Error(`Cannot spring ${typeof n} values`);
  }
}
function di(e, t = {}) {
  const n = zt(e), { stiffness: r = 0.15, damping: i = 0.8, precision: l = 0.01 } = t;
  let s, a, o, u = e, f = e, c = 1, h = 0, _ = !1;
  function d(S, b = {}) {
    f = S;
    const T = o = {};
    return e == null || b.hard || g.stiffness >= 1 && g.damping >= 1 ? (_ = !0, s = hi(), u = S, n.set(e = f), Promise.resolve()) : (b.soft && (h = 1 / ((b.soft === !0 ? 0.5 : +b.soft) * 60), c = 0), a || (s = hi(), _ = !1, a = Yu((y) => {
      if (_)
        return _ = !1, a = null, !1;
      c = Math.min(c + h, 1);
      const m = {
        inv_mass: c,
        opts: g,
        settled: !0,
        dt: (y - s) * 60 / 1e3
      }, v = yr(m, u, e, f);
      return s = y, u = e, n.set(e = v), m.settled && (a = null), !m.settled;
    })), new Promise((y) => {
      a.promise.then(() => {
        T === o && y();
      });
    }));
  }
  const g = {
    set: d,
    update: (S, b) => d(S(f, e), b),
    subscribe: n.subscribe,
    stiffness: r,
    damping: i,
    precision: l
  };
  return g;
}
function Ku(e) {
  return e && e.__esModule && Object.prototype.hasOwnProperty.call(e, "default") ? e.default : e;
}
var $u = function(t) {
  return ef(t) && !tf(t);
};
function ef(e) {
  return !!e && typeof e == "object";
}
function tf(e) {
  var t = Object.prototype.toString.call(e);
  return t === "[object RegExp]" || t === "[object Date]" || lf(e);
}
var nf = typeof Symbol == "function" && Symbol.for, rf = nf ? Symbol.for("react.element") : 60103;
function lf(e) {
  return e.$$typeof === rf;
}
function sf(e) {
  return Array.isArray(e) ? [] : {};
}
function Ft(e, t) {
  return t.clone !== !1 && t.isMergeableObject(e) ? pt(sf(e), e, t) : e;
}
function of(e, t, n) {
  return e.concat(t).map(function(r) {
    return Ft(r, n);
  });
}
function af(e, t) {
  if (!t.customMerge)
    return pt;
  var n = t.customMerge(e);
  return typeof n == "function" ? n : pt;
}
function uf(e) {
  return Object.getOwnPropertySymbols ? Object.getOwnPropertySymbols(e).filter(function(t) {
    return Object.propertyIsEnumerable.call(e, t);
  }) : [];
}
function mi(e) {
  return Object.keys(e).concat(uf(e));
}
function Il(e, t) {
  try {
    return t in e;
  } catch {
    return !1;
  }
}
function ff(e, t) {
  return Il(e, t) && !(Object.hasOwnProperty.call(e, t) && Object.propertyIsEnumerable.call(e, t));
}
function cf(e, t, n) {
  var r = {};
  return n.isMergeableObject(e) && mi(e).forEach(function(i) {
    r[i] = Ft(e[i], n);
  }), mi(t).forEach(function(i) {
    ff(e, i) || (Il(e, i) && n.isMergeableObject(t[i]) ? r[i] = af(i, n)(e[i], t[i], n) : r[i] = Ft(t[i], n));
  }), r;
}
function pt(e, t, n) {
  n = n || {}, n.arrayMerge = n.arrayMerge || of, n.isMergeableObject = n.isMergeableObject || $u, n.cloneUnlessOtherwiseSpecified = Ft;
  var r = Array.isArray(t), i = Array.isArray(e), l = r === i;
  return l ? r ? n.arrayMerge(e, t, n) : cf(e, t, n) : Ft(t, n);
}
pt.all = function(t, n) {
  if (!Array.isArray(t))
    throw new Error("first argument should be an array");
  return t.reduce(function(r, i) {
    return pt(r, i, n);
  }, {});
};
var hf = pt, _f = hf;
const df = /* @__PURE__ */ Ku(_f);
var Er = function(e, t) {
  return Er = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(n, r) {
    n.__proto__ = r;
  } || function(n, r) {
    for (var i in r)
      Object.prototype.hasOwnProperty.call(r, i) && (n[i] = r[i]);
  }, Er(e, t);
};
function vn(e, t) {
  if (typeof t != "function" && t !== null)
    throw new TypeError("Class extends value " + String(t) + " is not a constructor or null");
  Er(e, t);
  function n() {
    this.constructor = e;
  }
  e.prototype = t === null ? Object.create(t) : (n.prototype = t.prototype, new n());
}
var U = function() {
  return U = Object.assign || function(t) {
    for (var n, r = 1, i = arguments.length; r < i; r++) {
      n = arguments[r];
      for (var l in n)
        Object.prototype.hasOwnProperty.call(n, l) && (t[l] = n[l]);
    }
    return t;
  }, U.apply(this, arguments);
};
function Xn(e, t, n) {
  if (n || arguments.length === 2)
    for (var r = 0, i = t.length, l; r < i; r++)
      (l || !(r in t)) && (l || (l = Array.prototype.slice.call(t, 0, r)), l[r] = t[r]);
  return e.concat(l || Array.prototype.slice.call(t));
}
var L;
(function(e) {
  e[e.EXPECT_ARGUMENT_CLOSING_BRACE = 1] = "EXPECT_ARGUMENT_CLOSING_BRACE", e[e.EMPTY_ARGUMENT = 2] = "EMPTY_ARGUMENT", e[e.MALFORMED_ARGUMENT = 3] = "MALFORMED_ARGUMENT", e[e.EXPECT_ARGUMENT_TYPE = 4] = "EXPECT_ARGUMENT_TYPE", e[e.INVALID_ARGUMENT_TYPE = 5] = "INVALID_ARGUMENT_TYPE", e[e.EXPECT_ARGUMENT_STYLE = 6] = "EXPECT_ARGUMENT_STYLE", e[e.INVALID_NUMBER_SKELETON = 7] = "INVALID_NUMBER_SKELETON", e[e.INVALID_DATE_TIME_SKELETON = 8] = "INVALID_DATE_TIME_SKELETON", e[e.EXPECT_NUMBER_SKELETON = 9] = "EXPECT_NUMBER_SKELETON", e[e.EXPECT_DATE_TIME_SKELETON = 10] = "EXPECT_DATE_TIME_SKELETON", e[e.UNCLOSED_QUOTE_IN_ARGUMENT_STYLE = 11] = "UNCLOSED_QUOTE_IN_ARGUMENT_STYLE", e[e.EXPECT_SELECT_ARGUMENT_OPTIONS = 12] = "EXPECT_SELECT_ARGUMENT_OPTIONS", e[e.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE = 13] = "EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE", e[e.INVALID_PLURAL_ARGUMENT_OFFSET_VALUE = 14] = "INVALID_PLURAL_ARGUMENT_OFFSET_VALUE", e[e.EXPECT_SELECT_ARGUMENT_SELECTOR = 15] = "EXPECT_SELECT_ARGUMENT_SELECTOR", e[e.EXPECT_PLURAL_ARGUMENT_SELECTOR = 16] = "EXPECT_PLURAL_ARGUMENT_SELECTOR", e[e.EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT = 17] = "EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT", e[e.EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT = 18] = "EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT", e[e.INVALID_PLURAL_ARGUMENT_SELECTOR = 19] = "INVALID_PLURAL_ARGUMENT_SELECTOR", e[e.DUPLICATE_PLURAL_ARGUMENT_SELECTOR = 20] = "DUPLICATE_PLURAL_ARGUMENT_SELECTOR", e[e.DUPLICATE_SELECT_ARGUMENT_SELECTOR = 21] = "DUPLICATE_SELECT_ARGUMENT_SELECTOR", e[e.MISSING_OTHER_CLAUSE = 22] = "MISSING_OTHER_CLAUSE", e[e.INVALID_TAG = 23] = "INVALID_TAG", e[e.INVALID_TAG_NAME = 25] = "INVALID_TAG_NAME", e[e.UNMATCHED_CLOSING_TAG = 26] = "UNMATCHED_CLOSING_TAG", e[e.UNCLOSED_TAG = 27] = "UNCLOSED_TAG";
})(L || (L = {}));
var z;
(function(e) {
  e[e.literal = 0] = "literal", e[e.argument = 1] = "argument", e[e.number = 2] = "number", e[e.date = 3] = "date", e[e.time = 4] = "time", e[e.select = 5] = "select", e[e.plural = 6] = "plural", e[e.pound = 7] = "pound", e[e.tag = 8] = "tag";
})(z || (z = {}));
var bt;
(function(e) {
  e[e.number = 0] = "number", e[e.dateTime = 1] = "dateTime";
})(bt || (bt = {}));
function gi(e) {
  return e.type === z.literal;
}
function mf(e) {
  return e.type === z.argument;
}
function Ll(e) {
  return e.type === z.number;
}
function Ol(e) {
  return e.type === z.date;
}
function Ml(e) {
  return e.type === z.time;
}
function Rl(e) {
  return e.type === z.select;
}
function Dl(e) {
  return e.type === z.plural;
}
function gf(e) {
  return e.type === z.pound;
}
function Ul(e) {
  return e.type === z.tag;
}
function Fl(e) {
  return !!(e && typeof e == "object" && e.type === bt.number);
}
function Sr(e) {
  return !!(e && typeof e == "object" && e.type === bt.dateTime);
}
var Gl = /[ \xA0\u1680\u2000-\u200A\u202F\u205F\u3000]/, pf = /(?:[Eec]{1,6}|G{1,5}|[Qq]{1,5}|(?:[yYur]+|U{1,5})|[ML]{1,5}|d{1,2}|D{1,3}|F{1}|[abB]{1,5}|[hkHK]{1,2}|w{1,2}|W{1}|m{1,2}|s{1,2}|[zZOvVxX]{1,4})(?=([^']*'[^']*')*[^']*$)/g;
function bf(e) {
  var t = {};
  return e.replace(pf, function(n) {
    var r = n.length;
    switch (n[0]) {
      case "G":
        t.era = r === 4 ? "long" : r === 5 ? "narrow" : "short";
        break;
      case "y":
        t.year = r === 2 ? "2-digit" : "numeric";
        break;
      case "Y":
      case "u":
      case "U":
      case "r":
        throw new RangeError("`Y/u/U/r` (year) patterns are not supported, use `y` instead");
      case "q":
      case "Q":
        throw new RangeError("`q/Q` (quarter) patterns are not supported");
      case "M":
      case "L":
        t.month = ["numeric", "2-digit", "short", "long", "narrow"][r - 1];
        break;
      case "w":
      case "W":
        throw new RangeError("`w/W` (week) patterns are not supported");
      case "d":
        t.day = ["numeric", "2-digit"][r - 1];
        break;
      case "D":
      case "F":
      case "g":
        throw new RangeError("`D/F/g` (day) patterns are not supported, use `d` instead");
      case "E":
        t.weekday = r === 4 ? "short" : r === 5 ? "narrow" : "short";
        break;
      case "e":
        if (r < 4)
          throw new RangeError("`e..eee` (weekday) patterns are not supported");
        t.weekday = ["short", "long", "narrow", "short"][r - 4];
        break;
      case "c":
        if (r < 4)
          throw new RangeError("`c..ccc` (weekday) patterns are not supported");
        t.weekday = ["short", "long", "narrow", "short"][r - 4];
        break;
      case "a":
        t.hour12 = !0;
        break;
      case "b":
      case "B":
        throw new RangeError("`b/B` (period) patterns are not supported, use `a` instead");
      case "h":
        t.hourCycle = "h12", t.hour = ["numeric", "2-digit"][r - 1];
        break;
      case "H":
        t.hourCycle = "h23", t.hour = ["numeric", "2-digit"][r - 1];
        break;
      case "K":
        t.hourCycle = "h11", t.hour = ["numeric", "2-digit"][r - 1];
        break;
      case "k":
        t.hourCycle = "h24", t.hour = ["numeric", "2-digit"][r - 1];
        break;
      case "j":
      case "J":
      case "C":
        throw new RangeError("`j/J/C` (hour) patterns are not supported, use `h/H/K/k` instead");
      case "m":
        t.minute = ["numeric", "2-digit"][r - 1];
        break;
      case "s":
        t.second = ["numeric", "2-digit"][r - 1];
        break;
      case "S":
      case "A":
        throw new RangeError("`S/A` (second) patterns are not supported, use `s` instead");
      case "z":
        t.timeZoneName = r < 4 ? "short" : "long";
        break;
      case "Z":
      case "O":
      case "v":
      case "V":
      case "X":
      case "x":
        throw new RangeError("`Z/O/v/V/X/x` (timeZone) patterns are not supported, use `z` instead");
    }
    return "";
  }), t;
}
var wf = /[\t-\r \x85\u200E\u200F\u2028\u2029]/i;
function vf(e) {
  if (e.length === 0)
    throw new Error("Number skeleton cannot be empty");
  for (var t = e.split(wf).filter(function(h) {
    return h.length > 0;
  }), n = [], r = 0, i = t; r < i.length; r++) {
    var l = i[r], s = l.split("/");
    if (s.length === 0)
      throw new Error("Invalid number skeleton");
    for (var a = s[0], o = s.slice(1), u = 0, f = o; u < f.length; u++) {
      var c = f[u];
      if (c.length === 0)
        throw new Error("Invalid number skeleton");
    }
    n.push({ stem: a, options: o });
  }
  return n;
}
function yf(e) {
  return e.replace(/^(.*?)-/, "");
}
var pi = /^\.(?:(0+)(\*)?|(#+)|(0+)(#+))$/g, ql = /^(@+)?(\+|#+)?[rs]?$/g, Ef = /(\*)(0+)|(#+)(0+)|(0+)/g, xl = /^(0+)$/;
function bi(e) {
  var t = {};
  return e[e.length - 1] === "r" ? t.roundingPriority = "morePrecision" : e[e.length - 1] === "s" && (t.roundingPriority = "lessPrecision"), e.replace(ql, function(n, r, i) {
    return typeof i != "string" ? (t.minimumSignificantDigits = r.length, t.maximumSignificantDigits = r.length) : i === "+" ? t.minimumSignificantDigits = r.length : r[0] === "#" ? t.maximumSignificantDigits = r.length : (t.minimumSignificantDigits = r.length, t.maximumSignificantDigits = r.length + (typeof i == "string" ? i.length : 0)), "";
  }), t;
}
function jl(e) {
  switch (e) {
    case "sign-auto":
      return {
        signDisplay: "auto"
      };
    case "sign-accounting":
    case "()":
      return {
        currencySign: "accounting"
      };
    case "sign-always":
    case "+!":
      return {
        signDisplay: "always"
      };
    case "sign-accounting-always":
    case "()!":
      return {
        signDisplay: "always",
        currencySign: "accounting"
      };
    case "sign-except-zero":
    case "+?":
      return {
        signDisplay: "exceptZero"
      };
    case "sign-accounting-except-zero":
    case "()?":
      return {
        signDisplay: "exceptZero",
        currencySign: "accounting"
      };
    case "sign-never":
    case "+_":
      return {
        signDisplay: "never"
      };
  }
}
function Sf(e) {
  var t;
  if (e[0] === "E" && e[1] === "E" ? (t = {
    notation: "engineering"
  }, e = e.slice(2)) : e[0] === "E" && (t = {
    notation: "scientific"
  }, e = e.slice(1)), t) {
    var n = e.slice(0, 2);
    if (n === "+!" ? (t.signDisplay = "always", e = e.slice(2)) : n === "+?" && (t.signDisplay = "exceptZero", e = e.slice(2)), !xl.test(e))
      throw new Error("Malformed concise eng/scientific notation");
    t.minimumIntegerDigits = e.length;
  }
  return t;
}
function wi(e) {
  var t = {}, n = jl(e);
  return n || t;
}
function kf(e) {
  for (var t = {}, n = 0, r = e; n < r.length; n++) {
    var i = r[n];
    switch (i.stem) {
      case "percent":
      case "%":
        t.style = "percent";
        continue;
      case "%x100":
        t.style = "percent", t.scale = 100;
        continue;
      case "currency":
        t.style = "currency", t.currency = i.options[0];
        continue;
      case "group-off":
      case ",_":
        t.useGrouping = !1;
        continue;
      case "precision-integer":
      case ".":
        t.maximumFractionDigits = 0;
        continue;
      case "measure-unit":
      case "unit":
        t.style = "unit", t.unit = yf(i.options[0]);
        continue;
      case "compact-short":
      case "K":
        t.notation = "compact", t.compactDisplay = "short";
        continue;
      case "compact-long":
      case "KK":
        t.notation = "compact", t.compactDisplay = "long";
        continue;
      case "scientific":
        t = U(U(U({}, t), { notation: "scientific" }), i.options.reduce(function(o, u) {
          return U(U({}, o), wi(u));
        }, {}));
        continue;
      case "engineering":
        t = U(U(U({}, t), { notation: "engineering" }), i.options.reduce(function(o, u) {
          return U(U({}, o), wi(u));
        }, {}));
        continue;
      case "notation-simple":
        t.notation = "standard";
        continue;
      case "unit-width-narrow":
        t.currencyDisplay = "narrowSymbol", t.unitDisplay = "narrow";
        continue;
      case "unit-width-short":
        t.currencyDisplay = "code", t.unitDisplay = "short";
        continue;
      case "unit-width-full-name":
        t.currencyDisplay = "name", t.unitDisplay = "long";
        continue;
      case "unit-width-iso-code":
        t.currencyDisplay = "symbol";
        continue;
      case "scale":
        t.scale = parseFloat(i.options[0]);
        continue;
      case "integer-width":
        if (i.options.length > 1)
          throw new RangeError("integer-width stems only accept a single optional option");
        i.options[0].replace(Ef, function(o, u, f, c, h, _) {
          if (u)
            t.minimumIntegerDigits = f.length;
          else {
            if (c && h)
              throw new Error("We currently do not support maximum integer digits");
            if (_)
              throw new Error("We currently do not support exact integer digits");
          }
          return "";
        });
        continue;
    }
    if (xl.test(i.stem)) {
      t.minimumIntegerDigits = i.stem.length;
      continue;
    }
    if (pi.test(i.stem)) {
      if (i.options.length > 1)
        throw new RangeError("Fraction-precision stems only accept a single optional option");
      i.stem.replace(pi, function(o, u, f, c, h, _) {
        return f === "*" ? t.minimumFractionDigits = u.length : c && c[0] === "#" ? t.maximumFractionDigits = c.length : h && _ ? (t.minimumFractionDigits = h.length, t.maximumFractionDigits = h.length + _.length) : (t.minimumFractionDigits = u.length, t.maximumFractionDigits = u.length), "";
      });
      var l = i.options[0];
      l === "w" ? t = U(U({}, t), { trailingZeroDisplay: "stripIfInteger" }) : l && (t = U(U({}, t), bi(l)));
      continue;
    }
    if (ql.test(i.stem)) {
      t = U(U({}, t), bi(i.stem));
      continue;
    }
    var s = jl(i.stem);
    s && (t = U(U({}, t), s));
    var a = Sf(i.stem);
    a && (t = U(U({}, t), a));
  }
  return t;
}
var en = {
  AX: [
    "H"
  ],
  BQ: [
    "H"
  ],
  CP: [
    "H"
  ],
  CZ: [
    "H"
  ],
  DK: [
    "H"
  ],
  FI: [
    "H"
  ],
  ID: [
    "H"
  ],
  IS: [
    "H"
  ],
  ML: [
    "H"
  ],
  NE: [
    "H"
  ],
  RU: [
    "H"
  ],
  SE: [
    "H"
  ],
  SJ: [
    "H"
  ],
  SK: [
    "H"
  ],
  AS: [
    "h",
    "H"
  ],
  BT: [
    "h",
    "H"
  ],
  DJ: [
    "h",
    "H"
  ],
  ER: [
    "h",
    "H"
  ],
  GH: [
    "h",
    "H"
  ],
  IN: [
    "h",
    "H"
  ],
  LS: [
    "h",
    "H"
  ],
  PG: [
    "h",
    "H"
  ],
  PW: [
    "h",
    "H"
  ],
  SO: [
    "h",
    "H"
  ],
  TO: [
    "h",
    "H"
  ],
  VU: [
    "h",
    "H"
  ],
  WS: [
    "h",
    "H"
  ],
  "001": [
    "H",
    "h"
  ],
  AL: [
    "h",
    "H",
    "hB"
  ],
  TD: [
    "h",
    "H",
    "hB"
  ],
  "ca-ES": [
    "H",
    "h",
    "hB"
  ],
  CF: [
    "H",
    "h",
    "hB"
  ],
  CM: [
    "H",
    "h",
    "hB"
  ],
  "fr-CA": [
    "H",
    "h",
    "hB"
  ],
  "gl-ES": [
    "H",
    "h",
    "hB"
  ],
  "it-CH": [
    "H",
    "h",
    "hB"
  ],
  "it-IT": [
    "H",
    "h",
    "hB"
  ],
  LU: [
    "H",
    "h",
    "hB"
  ],
  NP: [
    "H",
    "h",
    "hB"
  ],
  PF: [
    "H",
    "h",
    "hB"
  ],
  SC: [
    "H",
    "h",
    "hB"
  ],
  SM: [
    "H",
    "h",
    "hB"
  ],
  SN: [
    "H",
    "h",
    "hB"
  ],
  TF: [
    "H",
    "h",
    "hB"
  ],
  VA: [
    "H",
    "h",
    "hB"
  ],
  CY: [
    "h",
    "H",
    "hb",
    "hB"
  ],
  GR: [
    "h",
    "H",
    "hb",
    "hB"
  ],
  CO: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  DO: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  KP: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  KR: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  NA: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  PA: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  PR: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  VE: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  AC: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  AI: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  BW: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  BZ: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  CC: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  CK: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  CX: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  DG: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  FK: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  GB: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  GG: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  GI: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  IE: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  IM: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  IO: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  JE: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  LT: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  MK: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  MN: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  MS: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  NF: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  NG: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  NR: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  NU: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  PN: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  SH: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  SX: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  TA: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  ZA: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  "af-ZA": [
    "H",
    "h",
    "hB",
    "hb"
  ],
  AR: [
    "H",
    "h",
    "hB",
    "hb"
  ],
  CL: [
    "H",
    "h",
    "hB",
    "hb"
  ],
  CR: [
    "H",
    "h",
    "hB",
    "hb"
  ],
  CU: [
    "H",
    "h",
    "hB",
    "hb"
  ],
  EA: [
    "H",
    "h",
    "hB",
    "hb"
  ],
  "es-BO": [
    "H",
    "h",
    "hB",
    "hb"
  ],
  "es-BR": [
    "H",
    "h",
    "hB",
    "hb"
  ],
  "es-EC": [
    "H",
    "h",
    "hB",
    "hb"
  ],
  "es-ES": [
    "H",
    "h",
    "hB",
    "hb"
  ],
  "es-GQ": [
    "H",
    "h",
    "hB",
    "hb"
  ],
  "es-PE": [
    "H",
    "h",
    "hB",
    "hb"
  ],
  GT: [
    "H",
    "h",
    "hB",
    "hb"
  ],
  HN: [
    "H",
    "h",
    "hB",
    "hb"
  ],
  IC: [
    "H",
    "h",
    "hB",
    "hb"
  ],
  KG: [
    "H",
    "h",
    "hB",
    "hb"
  ],
  KM: [
    "H",
    "h",
    "hB",
    "hb"
  ],
  LK: [
    "H",
    "h",
    "hB",
    "hb"
  ],
  MA: [
    "H",
    "h",
    "hB",
    "hb"
  ],
  MX: [
    "H",
    "h",
    "hB",
    "hb"
  ],
  NI: [
    "H",
    "h",
    "hB",
    "hb"
  ],
  PY: [
    "H",
    "h",
    "hB",
    "hb"
  ],
  SV: [
    "H",
    "h",
    "hB",
    "hb"
  ],
  UY: [
    "H",
    "h",
    "hB",
    "hb"
  ],
  JP: [
    "H",
    "h",
    "K"
  ],
  AD: [
    "H",
    "hB"
  ],
  AM: [
    "H",
    "hB"
  ],
  AO: [
    "H",
    "hB"
  ],
  AT: [
    "H",
    "hB"
  ],
  AW: [
    "H",
    "hB"
  ],
  BE: [
    "H",
    "hB"
  ],
  BF: [
    "H",
    "hB"
  ],
  BJ: [
    "H",
    "hB"
  ],
  BL: [
    "H",
    "hB"
  ],
  BR: [
    "H",
    "hB"
  ],
  CG: [
    "H",
    "hB"
  ],
  CI: [
    "H",
    "hB"
  ],
  CV: [
    "H",
    "hB"
  ],
  DE: [
    "H",
    "hB"
  ],
  EE: [
    "H",
    "hB"
  ],
  FR: [
    "H",
    "hB"
  ],
  GA: [
    "H",
    "hB"
  ],
  GF: [
    "H",
    "hB"
  ],
  GN: [
    "H",
    "hB"
  ],
  GP: [
    "H",
    "hB"
  ],
  GW: [
    "H",
    "hB"
  ],
  HR: [
    "H",
    "hB"
  ],
  IL: [
    "H",
    "hB"
  ],
  IT: [
    "H",
    "hB"
  ],
  KZ: [
    "H",
    "hB"
  ],
  MC: [
    "H",
    "hB"
  ],
  MD: [
    "H",
    "hB"
  ],
  MF: [
    "H",
    "hB"
  ],
  MQ: [
    "H",
    "hB"
  ],
  MZ: [
    "H",
    "hB"
  ],
  NC: [
    "H",
    "hB"
  ],
  NL: [
    "H",
    "hB"
  ],
  PM: [
    "H",
    "hB"
  ],
  PT: [
    "H",
    "hB"
  ],
  RE: [
    "H",
    "hB"
  ],
  RO: [
    "H",
    "hB"
  ],
  SI: [
    "H",
    "hB"
  ],
  SR: [
    "H",
    "hB"
  ],
  ST: [
    "H",
    "hB"
  ],
  TG: [
    "H",
    "hB"
  ],
  TR: [
    "H",
    "hB"
  ],
  WF: [
    "H",
    "hB"
  ],
  YT: [
    "H",
    "hB"
  ],
  BD: [
    "h",
    "hB",
    "H"
  ],
  PK: [
    "h",
    "hB",
    "H"
  ],
  AZ: [
    "H",
    "hB",
    "h"
  ],
  BA: [
    "H",
    "hB",
    "h"
  ],
  BG: [
    "H",
    "hB",
    "h"
  ],
  CH: [
    "H",
    "hB",
    "h"
  ],
  GE: [
    "H",
    "hB",
    "h"
  ],
  LI: [
    "H",
    "hB",
    "h"
  ],
  ME: [
    "H",
    "hB",
    "h"
  ],
  RS: [
    "H",
    "hB",
    "h"
  ],
  UA: [
    "H",
    "hB",
    "h"
  ],
  UZ: [
    "H",
    "hB",
    "h"
  ],
  XK: [
    "H",
    "hB",
    "h"
  ],
  AG: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  AU: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  BB: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  BM: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  BS: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  CA: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  DM: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  "en-001": [
    "h",
    "hb",
    "H",
    "hB"
  ],
  FJ: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  FM: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  GD: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  GM: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  GU: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  GY: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  JM: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  KI: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  KN: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  KY: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  LC: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  LR: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  MH: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  MP: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  MW: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  NZ: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  SB: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  SG: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  SL: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  SS: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  SZ: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  TC: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  TT: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  UM: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  US: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  VC: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  VG: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  VI: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  ZM: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  BO: [
    "H",
    "hB",
    "h",
    "hb"
  ],
  EC: [
    "H",
    "hB",
    "h",
    "hb"
  ],
  ES: [
    "H",
    "hB",
    "h",
    "hb"
  ],
  GQ: [
    "H",
    "hB",
    "h",
    "hb"
  ],
  PE: [
    "H",
    "hB",
    "h",
    "hb"
  ],
  AE: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  "ar-001": [
    "h",
    "hB",
    "hb",
    "H"
  ],
  BH: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  DZ: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  EG: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  EH: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  HK: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  IQ: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  JO: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  KW: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  LB: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  LY: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  MO: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  MR: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  OM: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  PH: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  PS: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  QA: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  SA: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  SD: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  SY: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  TN: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  YE: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  AF: [
    "H",
    "hb",
    "hB",
    "h"
  ],
  LA: [
    "H",
    "hb",
    "hB",
    "h"
  ],
  CN: [
    "H",
    "hB",
    "hb",
    "h"
  ],
  LV: [
    "H",
    "hB",
    "hb",
    "h"
  ],
  TL: [
    "H",
    "hB",
    "hb",
    "h"
  ],
  "zu-ZA": [
    "H",
    "hB",
    "hb",
    "h"
  ],
  CD: [
    "hB",
    "H"
  ],
  IR: [
    "hB",
    "H"
  ],
  "hi-IN": [
    "hB",
    "h",
    "H"
  ],
  "kn-IN": [
    "hB",
    "h",
    "H"
  ],
  "ml-IN": [
    "hB",
    "h",
    "H"
  ],
  "te-IN": [
    "hB",
    "h",
    "H"
  ],
  KH: [
    "hB",
    "h",
    "H",
    "hb"
  ],
  "ta-IN": [
    "hB",
    "h",
    "hb",
    "H"
  ],
  BN: [
    "hb",
    "hB",
    "h",
    "H"
  ],
  MY: [
    "hb",
    "hB",
    "h",
    "H"
  ],
  ET: [
    "hB",
    "hb",
    "h",
    "H"
  ],
  "gu-IN": [
    "hB",
    "hb",
    "h",
    "H"
  ],
  "mr-IN": [
    "hB",
    "hb",
    "h",
    "H"
  ],
  "pa-IN": [
    "hB",
    "hb",
    "h",
    "H"
  ],
  TW: [
    "hB",
    "hb",
    "h",
    "H"
  ],
  KE: [
    "hB",
    "hb",
    "H",
    "h"
  ],
  MM: [
    "hB",
    "hb",
    "H",
    "h"
  ],
  TZ: [
    "hB",
    "hb",
    "H",
    "h"
  ],
  UG: [
    "hB",
    "hb",
    "H",
    "h"
  ]
};
function Tf(e, t) {
  for (var n = "", r = 0; r < e.length; r++) {
    var i = e.charAt(r);
    if (i === "j") {
      for (var l = 0; r + 1 < e.length && e.charAt(r + 1) === i; )
        l++, r++;
      var s = 1 + (l & 1), a = l < 2 ? 1 : 3 + (l >> 1), o = "a", u = Bf(t);
      for ((u == "H" || u == "k") && (a = 0); a-- > 0; )
        n += o;
      for (; s-- > 0; )
        n = u + n;
    } else
      i === "J" ? n += "H" : n += i;
  }
  return n;
}
function Bf(e) {
  var t = e.hourCycle;
  if (t === void 0 && // @ts-ignore hourCycle(s) is not identified yet
  e.hourCycles && // @ts-ignore
  e.hourCycles.length && (t = e.hourCycles[0]), t)
    switch (t) {
      case "h24":
        return "k";
      case "h23":
        return "H";
      case "h12":
        return "h";
      case "h11":
        return "K";
      default:
        throw new Error("Invalid hourCycle");
    }
  var n = e.language, r;
  n !== "root" && (r = e.maximize().region);
  var i = en[r || ""] || en[n || ""] || en["".concat(n, "-001")] || en["001"];
  return i[0];
}
var Zn, Af = new RegExp("^".concat(Gl.source, "*")), Cf = new RegExp("".concat(Gl.source, "*$"));
function O(e, t) {
  return { start: e, end: t };
}
var Hf = !!String.prototype.startsWith, Pf = !!String.fromCodePoint, Nf = !!Object.fromEntries, If = !!String.prototype.codePointAt, Lf = !!String.prototype.trimStart, Of = !!String.prototype.trimEnd, Mf = !!Number.isSafeInteger, Rf = Mf ? Number.isSafeInteger : function(e) {
  return typeof e == "number" && isFinite(e) && Math.floor(e) === e && Math.abs(e) <= 9007199254740991;
}, kr = !0;
try {
  var Df = Vl("([^\\p{White_Space}\\p{Pattern_Syntax}]*)", "yu");
  kr = ((Zn = Df.exec("a")) === null || Zn === void 0 ? void 0 : Zn[0]) === "a";
} catch {
  kr = !1;
}
var vi = Hf ? (
  // Native
  function(t, n, r) {
    return t.startsWith(n, r);
  }
) : (
  // For IE11
  function(t, n, r) {
    return t.slice(r, r + n.length) === n;
  }
), Tr = Pf ? String.fromCodePoint : (
  // IE11
  function() {
    for (var t = [], n = 0; n < arguments.length; n++)
      t[n] = arguments[n];
    for (var r = "", i = t.length, l = 0, s; i > l; ) {
      if (s = t[l++], s > 1114111)
        throw RangeError(s + " is not a valid code point");
      r += s < 65536 ? String.fromCharCode(s) : String.fromCharCode(((s -= 65536) >> 10) + 55296, s % 1024 + 56320);
    }
    return r;
  }
), yi = (
  // native
  Nf ? Object.fromEntries : (
    // Ponyfill
    function(t) {
      for (var n = {}, r = 0, i = t; r < i.length; r++) {
        var l = i[r], s = l[0], a = l[1];
        n[s] = a;
      }
      return n;
    }
  )
), zl = If ? (
  // Native
  function(t, n) {
    return t.codePointAt(n);
  }
) : (
  // IE 11
  function(t, n) {
    var r = t.length;
    if (!(n < 0 || n >= r)) {
      var i = t.charCodeAt(n), l;
      return i < 55296 || i > 56319 || n + 1 === r || (l = t.charCodeAt(n + 1)) < 56320 || l > 57343 ? i : (i - 55296 << 10) + (l - 56320) + 65536;
    }
  }
), Uf = Lf ? (
  // Native
  function(t) {
    return t.trimStart();
  }
) : (
  // Ponyfill
  function(t) {
    return t.replace(Af, "");
  }
), Ff = Of ? (
  // Native
  function(t) {
    return t.trimEnd();
  }
) : (
  // Ponyfill
  function(t) {
    return t.replace(Cf, "");
  }
);
function Vl(e, t) {
  return new RegExp(e, t);
}
var Br;
if (kr) {
  var Ei = Vl("([^\\p{White_Space}\\p{Pattern_Syntax}]*)", "yu");
  Br = function(t, n) {
    var r;
    Ei.lastIndex = n;
    var i = Ei.exec(t);
    return (r = i[1]) !== null && r !== void 0 ? r : "";
  };
} else
  Br = function(t, n) {
    for (var r = []; ; ) {
      var i = zl(t, n);
      if (i === void 0 || Xl(i) || jf(i))
        break;
      r.push(i), n += i >= 65536 ? 2 : 1;
    }
    return Tr.apply(void 0, r);
  };
var Gf = (
  /** @class */
  function() {
    function e(t, n) {
      n === void 0 && (n = {}), this.message = t, this.position = { offset: 0, line: 1, column: 1 }, this.ignoreTag = !!n.ignoreTag, this.locale = n.locale, this.requiresOtherClause = !!n.requiresOtherClause, this.shouldParseSkeletons = !!n.shouldParseSkeletons;
    }
    return e.prototype.parse = function() {
      if (this.offset() !== 0)
        throw Error("parser can only be used once");
      return this.parseMessage(0, "", !1);
    }, e.prototype.parseMessage = function(t, n, r) {
      for (var i = []; !this.isEOF(); ) {
        var l = this.char();
        if (l === 123) {
          var s = this.parseArgument(t, r);
          if (s.err)
            return s;
          i.push(s.val);
        } else {
          if (l === 125 && t > 0)
            break;
          if (l === 35 && (n === "plural" || n === "selectordinal")) {
            var a = this.clonePosition();
            this.bump(), i.push({
              type: z.pound,
              location: O(a, this.clonePosition())
            });
          } else if (l === 60 && !this.ignoreTag && this.peek() === 47) {
            if (r)
              break;
            return this.error(L.UNMATCHED_CLOSING_TAG, O(this.clonePosition(), this.clonePosition()));
          } else if (l === 60 && !this.ignoreTag && Ar(this.peek() || 0)) {
            var s = this.parseTag(t, n);
            if (s.err)
              return s;
            i.push(s.val);
          } else {
            var s = this.parseLiteral(t, n);
            if (s.err)
              return s;
            i.push(s.val);
          }
        }
      }
      return { val: i, err: null };
    }, e.prototype.parseTag = function(t, n) {
      var r = this.clonePosition();
      this.bump();
      var i = this.parseTagName();
      if (this.bumpSpace(), this.bumpIf("/>"))
        return {
          val: {
            type: z.literal,
            value: "<".concat(i, "/>"),
            location: O(r, this.clonePosition())
          },
          err: null
        };
      if (this.bumpIf(">")) {
        var l = this.parseMessage(t + 1, n, !0);
        if (l.err)
          return l;
        var s = l.val, a = this.clonePosition();
        if (this.bumpIf("</")) {
          if (this.isEOF() || !Ar(this.char()))
            return this.error(L.INVALID_TAG, O(a, this.clonePosition()));
          var o = this.clonePosition(), u = this.parseTagName();
          return i !== u ? this.error(L.UNMATCHED_CLOSING_TAG, O(o, this.clonePosition())) : (this.bumpSpace(), this.bumpIf(">") ? {
            val: {
              type: z.tag,
              value: i,
              children: s,
              location: O(r, this.clonePosition())
            },
            err: null
          } : this.error(L.INVALID_TAG, O(a, this.clonePosition())));
        } else
          return this.error(L.UNCLOSED_TAG, O(r, this.clonePosition()));
      } else
        return this.error(L.INVALID_TAG, O(r, this.clonePosition()));
    }, e.prototype.parseTagName = function() {
      var t = this.offset();
      for (this.bump(); !this.isEOF() && xf(this.char()); )
        this.bump();
      return this.message.slice(t, this.offset());
    }, e.prototype.parseLiteral = function(t, n) {
      for (var r = this.clonePosition(), i = ""; ; ) {
        var l = this.tryParseQuote(n);
        if (l) {
          i += l;
          continue;
        }
        var s = this.tryParseUnquoted(t, n);
        if (s) {
          i += s;
          continue;
        }
        var a = this.tryParseLeftAngleBracket();
        if (a) {
          i += a;
          continue;
        }
        break;
      }
      var o = O(r, this.clonePosition());
      return {
        val: { type: z.literal, value: i, location: o },
        err: null
      };
    }, e.prototype.tryParseLeftAngleBracket = function() {
      return !this.isEOF() && this.char() === 60 && (this.ignoreTag || // If at the opening tag or closing tag position, bail.
      !qf(this.peek() || 0)) ? (this.bump(), "<") : null;
    }, e.prototype.tryParseQuote = function(t) {
      if (this.isEOF() || this.char() !== 39)
        return null;
      switch (this.peek()) {
        case 39:
          return this.bump(), this.bump(), "'";
        case 123:
        case 60:
        case 62:
        case 125:
          break;
        case 35:
          if (t === "plural" || t === "selectordinal")
            break;
          return null;
        default:
          return null;
      }
      this.bump();
      var n = [this.char()];
      for (this.bump(); !this.isEOF(); ) {
        var r = this.char();
        if (r === 39)
          if (this.peek() === 39)
            n.push(39), this.bump();
          else {
            this.bump();
            break;
          }
        else
          n.push(r);
        this.bump();
      }
      return Tr.apply(void 0, n);
    }, e.prototype.tryParseUnquoted = function(t, n) {
      if (this.isEOF())
        return null;
      var r = this.char();
      return r === 60 || r === 123 || r === 35 && (n === "plural" || n === "selectordinal") || r === 125 && t > 0 ? null : (this.bump(), Tr(r));
    }, e.prototype.parseArgument = function(t, n) {
      var r = this.clonePosition();
      if (this.bump(), this.bumpSpace(), this.isEOF())
        return this.error(L.EXPECT_ARGUMENT_CLOSING_BRACE, O(r, this.clonePosition()));
      if (this.char() === 125)
        return this.bump(), this.error(L.EMPTY_ARGUMENT, O(r, this.clonePosition()));
      var i = this.parseIdentifierIfPossible().value;
      if (!i)
        return this.error(L.MALFORMED_ARGUMENT, O(r, this.clonePosition()));
      if (this.bumpSpace(), this.isEOF())
        return this.error(L.EXPECT_ARGUMENT_CLOSING_BRACE, O(r, this.clonePosition()));
      switch (this.char()) {
        case 125:
          return this.bump(), {
            val: {
              type: z.argument,
              // value does not include the opening and closing braces.
              value: i,
              location: O(r, this.clonePosition())
            },
            err: null
          };
        case 44:
          return this.bump(), this.bumpSpace(), this.isEOF() ? this.error(L.EXPECT_ARGUMENT_CLOSING_BRACE, O(r, this.clonePosition())) : this.parseArgumentOptions(t, n, i, r);
        default:
          return this.error(L.MALFORMED_ARGUMENT, O(r, this.clonePosition()));
      }
    }, e.prototype.parseIdentifierIfPossible = function() {
      var t = this.clonePosition(), n = this.offset(), r = Br(this.message, n), i = n + r.length;
      this.bumpTo(i);
      var l = this.clonePosition(), s = O(t, l);
      return { value: r, location: s };
    }, e.prototype.parseArgumentOptions = function(t, n, r, i) {
      var l, s = this.clonePosition(), a = this.parseIdentifierIfPossible().value, o = this.clonePosition();
      switch (a) {
        case "":
          return this.error(L.EXPECT_ARGUMENT_TYPE, O(s, o));
        case "number":
        case "date":
        case "time": {
          this.bumpSpace();
          var u = null;
          if (this.bumpIf(",")) {
            this.bumpSpace();
            var f = this.clonePosition(), c = this.parseSimpleArgStyleIfPossible();
            if (c.err)
              return c;
            var h = Ff(c.val);
            if (h.length === 0)
              return this.error(L.EXPECT_ARGUMENT_STYLE, O(this.clonePosition(), this.clonePosition()));
            var _ = O(f, this.clonePosition());
            u = { style: h, styleLocation: _ };
          }
          var d = this.tryParseArgumentClose(i);
          if (d.err)
            return d;
          var g = O(i, this.clonePosition());
          if (u && vi(u == null ? void 0 : u.style, "::", 0)) {
            var S = Uf(u.style.slice(2));
            if (a === "number") {
              var c = this.parseNumberSkeletonFromString(S, u.styleLocation);
              return c.err ? c : {
                val: { type: z.number, value: r, location: g, style: c.val },
                err: null
              };
            } else {
              if (S.length === 0)
                return this.error(L.EXPECT_DATE_TIME_SKELETON, g);
              var b = S;
              this.locale && (b = Tf(S, this.locale));
              var h = {
                type: bt.dateTime,
                pattern: b,
                location: u.styleLocation,
                parsedOptions: this.shouldParseSkeletons ? bf(b) : {}
              }, T = a === "date" ? z.date : z.time;
              return {
                val: { type: T, value: r, location: g, style: h },
                err: null
              };
            }
          }
          return {
            val: {
              type: a === "number" ? z.number : a === "date" ? z.date : z.time,
              value: r,
              location: g,
              style: (l = u == null ? void 0 : u.style) !== null && l !== void 0 ? l : null
            },
            err: null
          };
        }
        case "plural":
        case "selectordinal":
        case "select": {
          var y = this.clonePosition();
          if (this.bumpSpace(), !this.bumpIf(","))
            return this.error(L.EXPECT_SELECT_ARGUMENT_OPTIONS, O(y, U({}, y)));
          this.bumpSpace();
          var m = this.parseIdentifierIfPossible(), v = 0;
          if (a !== "select" && m.value === "offset") {
            if (!this.bumpIf(":"))
              return this.error(L.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE, O(this.clonePosition(), this.clonePosition()));
            this.bumpSpace();
            var c = this.tryParseDecimalInteger(L.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE, L.INVALID_PLURAL_ARGUMENT_OFFSET_VALUE);
            if (c.err)
              return c;
            this.bumpSpace(), m = this.parseIdentifierIfPossible(), v = c.val;
          }
          var G = this.tryParsePluralOrSelectOptions(t, a, n, m);
          if (G.err)
            return G;
          var d = this.tryParseArgumentClose(i);
          if (d.err)
            return d;
          var F = O(i, this.clonePosition());
          return a === "select" ? {
            val: {
              type: z.select,
              value: r,
              options: yi(G.val),
              location: F
            },
            err: null
          } : {
            val: {
              type: z.plural,
              value: r,
              options: yi(G.val),
              offset: v,
              pluralType: a === "plural" ? "cardinal" : "ordinal",
              location: F
            },
            err: null
          };
        }
        default:
          return this.error(L.INVALID_ARGUMENT_TYPE, O(s, o));
      }
    }, e.prototype.tryParseArgumentClose = function(t) {
      return this.isEOF() || this.char() !== 125 ? this.error(L.EXPECT_ARGUMENT_CLOSING_BRACE, O(t, this.clonePosition())) : (this.bump(), { val: !0, err: null });
    }, e.prototype.parseSimpleArgStyleIfPossible = function() {
      for (var t = 0, n = this.clonePosition(); !this.isEOF(); ) {
        var r = this.char();
        switch (r) {
          case 39: {
            this.bump();
            var i = this.clonePosition();
            if (!this.bumpUntil("'"))
              return this.error(L.UNCLOSED_QUOTE_IN_ARGUMENT_STYLE, O(i, this.clonePosition()));
            this.bump();
            break;
          }
          case 123: {
            t += 1, this.bump();
            break;
          }
          case 125: {
            if (t > 0)
              t -= 1;
            else
              return {
                val: this.message.slice(n.offset, this.offset()),
                err: null
              };
            break;
          }
          default:
            this.bump();
            break;
        }
      }
      return {
        val: this.message.slice(n.offset, this.offset()),
        err: null
      };
    }, e.prototype.parseNumberSkeletonFromString = function(t, n) {
      var r = [];
      try {
        r = vf(t);
      } catch {
        return this.error(L.INVALID_NUMBER_SKELETON, n);
      }
      return {
        val: {
          type: bt.number,
          tokens: r,
          location: n,
          parsedOptions: this.shouldParseSkeletons ? kf(r) : {}
        },
        err: null
      };
    }, e.prototype.tryParsePluralOrSelectOptions = function(t, n, r, i) {
      for (var l, s = !1, a = [], o = /* @__PURE__ */ new Set(), u = i.value, f = i.location; ; ) {
        if (u.length === 0) {
          var c = this.clonePosition();
          if (n !== "select" && this.bumpIf("=")) {
            var h = this.tryParseDecimalInteger(L.EXPECT_PLURAL_ARGUMENT_SELECTOR, L.INVALID_PLURAL_ARGUMENT_SELECTOR);
            if (h.err)
              return h;
            f = O(c, this.clonePosition()), u = this.message.slice(c.offset, this.offset());
          } else
            break;
        }
        if (o.has(u))
          return this.error(n === "select" ? L.DUPLICATE_SELECT_ARGUMENT_SELECTOR : L.DUPLICATE_PLURAL_ARGUMENT_SELECTOR, f);
        u === "other" && (s = !0), this.bumpSpace();
        var _ = this.clonePosition();
        if (!this.bumpIf("{"))
          return this.error(n === "select" ? L.EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT : L.EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT, O(this.clonePosition(), this.clonePosition()));
        var d = this.parseMessage(t + 1, n, r);
        if (d.err)
          return d;
        var g = this.tryParseArgumentClose(_);
        if (g.err)
          return g;
        a.push([
          u,
          {
            value: d.val,
            location: O(_, this.clonePosition())
          }
        ]), o.add(u), this.bumpSpace(), l = this.parseIdentifierIfPossible(), u = l.value, f = l.location;
      }
      return a.length === 0 ? this.error(n === "select" ? L.EXPECT_SELECT_ARGUMENT_SELECTOR : L.EXPECT_PLURAL_ARGUMENT_SELECTOR, O(this.clonePosition(), this.clonePosition())) : this.requiresOtherClause && !s ? this.error(L.MISSING_OTHER_CLAUSE, O(this.clonePosition(), this.clonePosition())) : { val: a, err: null };
    }, e.prototype.tryParseDecimalInteger = function(t, n) {
      var r = 1, i = this.clonePosition();
      this.bumpIf("+") || this.bumpIf("-") && (r = -1);
      for (var l = !1, s = 0; !this.isEOF(); ) {
        var a = this.char();
        if (a >= 48 && a <= 57)
          l = !0, s = s * 10 + (a - 48), this.bump();
        else
          break;
      }
      var o = O(i, this.clonePosition());
      return l ? (s *= r, Rf(s) ? { val: s, err: null } : this.error(n, o)) : this.error(t, o);
    }, e.prototype.offset = function() {
      return this.position.offset;
    }, e.prototype.isEOF = function() {
      return this.offset() === this.message.length;
    }, e.prototype.clonePosition = function() {
      return {
        offset: this.position.offset,
        line: this.position.line,
        column: this.position.column
      };
    }, e.prototype.char = function() {
      var t = this.position.offset;
      if (t >= this.message.length)
        throw Error("out of bound");
      var n = zl(this.message, t);
      if (n === void 0)
        throw Error("Offset ".concat(t, " is at invalid UTF-16 code unit boundary"));
      return n;
    }, e.prototype.error = function(t, n) {
      return {
        val: null,
        err: {
          kind: t,
          message: this.message,
          location: n
        }
      };
    }, e.prototype.bump = function() {
      if (!this.isEOF()) {
        var t = this.char();
        t === 10 ? (this.position.line += 1, this.position.column = 1, this.position.offset += 1) : (this.position.column += 1, this.position.offset += t < 65536 ? 1 : 2);
      }
    }, e.prototype.bumpIf = function(t) {
      if (vi(this.message, t, this.offset())) {
        for (var n = 0; n < t.length; n++)
          this.bump();
        return !0;
      }
      return !1;
    }, e.prototype.bumpUntil = function(t) {
      var n = this.offset(), r = this.message.indexOf(t, n);
      return r >= 0 ? (this.bumpTo(r), !0) : (this.bumpTo(this.message.length), !1);
    }, e.prototype.bumpTo = function(t) {
      if (this.offset() > t)
        throw Error("targetOffset ".concat(t, " must be greater than or equal to the current offset ").concat(this.offset()));
      for (t = Math.min(t, this.message.length); ; ) {
        var n = this.offset();
        if (n === t)
          break;
        if (n > t)
          throw Error("targetOffset ".concat(t, " is at invalid UTF-16 code unit boundary"));
        if (this.bump(), this.isEOF())
          break;
      }
    }, e.prototype.bumpSpace = function() {
      for (; !this.isEOF() && Xl(this.char()); )
        this.bump();
    }, e.prototype.peek = function() {
      if (this.isEOF())
        return null;
      var t = this.char(), n = this.offset(), r = this.message.charCodeAt(n + (t >= 65536 ? 2 : 1));
      return r ?? null;
    }, e;
  }()
);
function Ar(e) {
  return e >= 97 && e <= 122 || e >= 65 && e <= 90;
}
function qf(e) {
  return Ar(e) || e === 47;
}
function xf(e) {
  return e === 45 || e === 46 || e >= 48 && e <= 57 || e === 95 || e >= 97 && e <= 122 || e >= 65 && e <= 90 || e == 183 || e >= 192 && e <= 214 || e >= 216 && e <= 246 || e >= 248 && e <= 893 || e >= 895 && e <= 8191 || e >= 8204 && e <= 8205 || e >= 8255 && e <= 8256 || e >= 8304 && e <= 8591 || e >= 11264 && e <= 12271 || e >= 12289 && e <= 55295 || e >= 63744 && e <= 64975 || e >= 65008 && e <= 65533 || e >= 65536 && e <= 983039;
}
function Xl(e) {
  return e >= 9 && e <= 13 || e === 32 || e === 133 || e >= 8206 && e <= 8207 || e === 8232 || e === 8233;
}
function jf(e) {
  return e >= 33 && e <= 35 || e === 36 || e >= 37 && e <= 39 || e === 40 || e === 41 || e === 42 || e === 43 || e === 44 || e === 45 || e >= 46 && e <= 47 || e >= 58 && e <= 59 || e >= 60 && e <= 62 || e >= 63 && e <= 64 || e === 91 || e === 92 || e === 93 || e === 94 || e === 96 || e === 123 || e === 124 || e === 125 || e === 126 || e === 161 || e >= 162 && e <= 165 || e === 166 || e === 167 || e === 169 || e === 171 || e === 172 || e === 174 || e === 176 || e === 177 || e === 182 || e === 187 || e === 191 || e === 215 || e === 247 || e >= 8208 && e <= 8213 || e >= 8214 && e <= 8215 || e === 8216 || e === 8217 || e === 8218 || e >= 8219 && e <= 8220 || e === 8221 || e === 8222 || e === 8223 || e >= 8224 && e <= 8231 || e >= 8240 && e <= 8248 || e === 8249 || e === 8250 || e >= 8251 && e <= 8254 || e >= 8257 && e <= 8259 || e === 8260 || e === 8261 || e === 8262 || e >= 8263 && e <= 8273 || e === 8274 || e === 8275 || e >= 8277 && e <= 8286 || e >= 8592 && e <= 8596 || e >= 8597 && e <= 8601 || e >= 8602 && e <= 8603 || e >= 8604 && e <= 8607 || e === 8608 || e >= 8609 && e <= 8610 || e === 8611 || e >= 8612 && e <= 8613 || e === 8614 || e >= 8615 && e <= 8621 || e === 8622 || e >= 8623 && e <= 8653 || e >= 8654 && e <= 8655 || e >= 8656 && e <= 8657 || e === 8658 || e === 8659 || e === 8660 || e >= 8661 && e <= 8691 || e >= 8692 && e <= 8959 || e >= 8960 && e <= 8967 || e === 8968 || e === 8969 || e === 8970 || e === 8971 || e >= 8972 && e <= 8991 || e >= 8992 && e <= 8993 || e >= 8994 && e <= 9e3 || e === 9001 || e === 9002 || e >= 9003 && e <= 9083 || e === 9084 || e >= 9085 && e <= 9114 || e >= 9115 && e <= 9139 || e >= 9140 && e <= 9179 || e >= 9180 && e <= 9185 || e >= 9186 && e <= 9254 || e >= 9255 && e <= 9279 || e >= 9280 && e <= 9290 || e >= 9291 && e <= 9311 || e >= 9472 && e <= 9654 || e === 9655 || e >= 9656 && e <= 9664 || e === 9665 || e >= 9666 && e <= 9719 || e >= 9720 && e <= 9727 || e >= 9728 && e <= 9838 || e === 9839 || e >= 9840 && e <= 10087 || e === 10088 || e === 10089 || e === 10090 || e === 10091 || e === 10092 || e === 10093 || e === 10094 || e === 10095 || e === 10096 || e === 10097 || e === 10098 || e === 10099 || e === 10100 || e === 10101 || e >= 10132 && e <= 10175 || e >= 10176 && e <= 10180 || e === 10181 || e === 10182 || e >= 10183 && e <= 10213 || e === 10214 || e === 10215 || e === 10216 || e === 10217 || e === 10218 || e === 10219 || e === 10220 || e === 10221 || e === 10222 || e === 10223 || e >= 10224 && e <= 10239 || e >= 10240 && e <= 10495 || e >= 10496 && e <= 10626 || e === 10627 || e === 10628 || e === 10629 || e === 10630 || e === 10631 || e === 10632 || e === 10633 || e === 10634 || e === 10635 || e === 10636 || e === 10637 || e === 10638 || e === 10639 || e === 10640 || e === 10641 || e === 10642 || e === 10643 || e === 10644 || e === 10645 || e === 10646 || e === 10647 || e === 10648 || e >= 10649 && e <= 10711 || e === 10712 || e === 10713 || e === 10714 || e === 10715 || e >= 10716 && e <= 10747 || e === 10748 || e === 10749 || e >= 10750 && e <= 11007 || e >= 11008 && e <= 11055 || e >= 11056 && e <= 11076 || e >= 11077 && e <= 11078 || e >= 11079 && e <= 11084 || e >= 11085 && e <= 11123 || e >= 11124 && e <= 11125 || e >= 11126 && e <= 11157 || e === 11158 || e >= 11159 && e <= 11263 || e >= 11776 && e <= 11777 || e === 11778 || e === 11779 || e === 11780 || e === 11781 || e >= 11782 && e <= 11784 || e === 11785 || e === 11786 || e === 11787 || e === 11788 || e === 11789 || e >= 11790 && e <= 11798 || e === 11799 || e >= 11800 && e <= 11801 || e === 11802 || e === 11803 || e === 11804 || e === 11805 || e >= 11806 && e <= 11807 || e === 11808 || e === 11809 || e === 11810 || e === 11811 || e === 11812 || e === 11813 || e === 11814 || e === 11815 || e === 11816 || e === 11817 || e >= 11818 && e <= 11822 || e === 11823 || e >= 11824 && e <= 11833 || e >= 11834 && e <= 11835 || e >= 11836 && e <= 11839 || e === 11840 || e === 11841 || e === 11842 || e >= 11843 && e <= 11855 || e >= 11856 && e <= 11857 || e === 11858 || e >= 11859 && e <= 11903 || e >= 12289 && e <= 12291 || e === 12296 || e === 12297 || e === 12298 || e === 12299 || e === 12300 || e === 12301 || e === 12302 || e === 12303 || e === 12304 || e === 12305 || e >= 12306 && e <= 12307 || e === 12308 || e === 12309 || e === 12310 || e === 12311 || e === 12312 || e === 12313 || e === 12314 || e === 12315 || e === 12316 || e === 12317 || e >= 12318 && e <= 12319 || e === 12320 || e === 12336 || e === 64830 || e === 64831 || e >= 65093 && e <= 65094;
}
function Cr(e) {
  e.forEach(function(t) {
    if (delete t.location, Rl(t) || Dl(t))
      for (var n in t.options)
        delete t.options[n].location, Cr(t.options[n].value);
    else
      Ll(t) && Fl(t.style) || (Ol(t) || Ml(t)) && Sr(t.style) ? delete t.style.location : Ul(t) && Cr(t.children);
  });
}
function zf(e, t) {
  t === void 0 && (t = {}), t = U({ shouldParseSkeletons: !0, requiresOtherClause: !0 }, t);
  var n = new Gf(e, t).parse();
  if (n.err) {
    var r = SyntaxError(L[n.err.kind]);
    throw r.location = n.err.location, r.originalMessage = n.err.message, r;
  }
  return t != null && t.captureLocation || Cr(n.val), n.val;
}
function Wn(e, t) {
  var n = t && t.cache ? t.cache : Yf, r = t && t.serializer ? t.serializer : Jf, i = t && t.strategy ? t.strategy : Xf;
  return i(e, {
    cache: n,
    serializer: r
  });
}
function Vf(e) {
  return e == null || typeof e == "number" || typeof e == "boolean";
}
function Zl(e, t, n, r) {
  var i = Vf(r) ? r : n(r), l = t.get(i);
  return typeof l > "u" && (l = e.call(this, r), t.set(i, l)), l;
}
function Wl(e, t, n) {
  var r = Array.prototype.slice.call(arguments, 3), i = n(r), l = t.get(i);
  return typeof l > "u" && (l = e.apply(this, r), t.set(i, l)), l;
}
function Mr(e, t, n, r, i) {
  return n.bind(t, e, r, i);
}
function Xf(e, t) {
  var n = e.length === 1 ? Zl : Wl;
  return Mr(e, this, n, t.cache.create(), t.serializer);
}
function Zf(e, t) {
  return Mr(e, this, Wl, t.cache.create(), t.serializer);
}
function Wf(e, t) {
  return Mr(e, this, Zl, t.cache.create(), t.serializer);
}
var Jf = function() {
  return JSON.stringify(arguments);
};
function Rr() {
  this.cache = /* @__PURE__ */ Object.create(null);
}
Rr.prototype.get = function(e) {
  return this.cache[e];
};
Rr.prototype.set = function(e, t) {
  this.cache[e] = t;
};
var Yf = {
  create: function() {
    return new Rr();
  }
}, Jn = {
  variadic: Zf,
  monadic: Wf
}, wt;
(function(e) {
  e.MISSING_VALUE = "MISSING_VALUE", e.INVALID_VALUE = "INVALID_VALUE", e.MISSING_INTL_API = "MISSING_INTL_API";
})(wt || (wt = {}));
var yn = (
  /** @class */
  function(e) {
    vn(t, e);
    function t(n, r, i) {
      var l = e.call(this, n) || this;
      return l.code = r, l.originalMessage = i, l;
    }
    return t.prototype.toString = function() {
      return "[formatjs Error: ".concat(this.code, "] ").concat(this.message);
    }, t;
  }(Error)
), Si = (
  /** @class */
  function(e) {
    vn(t, e);
    function t(n, r, i, l) {
      return e.call(this, 'Invalid values for "'.concat(n, '": "').concat(r, '". Options are "').concat(Object.keys(i).join('", "'), '"'), wt.INVALID_VALUE, l) || this;
    }
    return t;
  }(yn)
), Qf = (
  /** @class */
  function(e) {
    vn(t, e);
    function t(n, r, i) {
      return e.call(this, 'Value for "'.concat(n, '" must be of type ').concat(r), wt.INVALID_VALUE, i) || this;
    }
    return t;
  }(yn)
), Kf = (
  /** @class */
  function(e) {
    vn(t, e);
    function t(n, r) {
      return e.call(this, 'The intl string context variable "'.concat(n, '" was not provided to the string "').concat(r, '"'), wt.MISSING_VALUE, r) || this;
    }
    return t;
  }(yn)
), se;
(function(e) {
  e[e.literal = 0] = "literal", e[e.object = 1] = "object";
})(se || (se = {}));
function $f(e) {
  return e.length < 2 ? e : e.reduce(function(t, n) {
    var r = t[t.length - 1];
    return !r || r.type !== se.literal || n.type !== se.literal ? t.push(n) : r.value += n.value, t;
  }, []);
}
function ec(e) {
  return typeof e == "function";
}
function fn(e, t, n, r, i, l, s) {
  if (e.length === 1 && gi(e[0]))
    return [
      {
        type: se.literal,
        value: e[0].value
      }
    ];
  for (var a = [], o = 0, u = e; o < u.length; o++) {
    var f = u[o];
    if (gi(f)) {
      a.push({
        type: se.literal,
        value: f.value
      });
      continue;
    }
    if (gf(f)) {
      typeof l == "number" && a.push({
        type: se.literal,
        value: n.getNumberFormat(t).format(l)
      });
      continue;
    }
    var c = f.value;
    if (!(i && c in i))
      throw new Kf(c, s);
    var h = i[c];
    if (mf(f)) {
      (!h || typeof h == "string" || typeof h == "number") && (h = typeof h == "string" || typeof h == "number" ? String(h) : ""), a.push({
        type: typeof h == "string" ? se.literal : se.object,
        value: h
      });
      continue;
    }
    if (Ol(f)) {
      var _ = typeof f.style == "string" ? r.date[f.style] : Sr(f.style) ? f.style.parsedOptions : void 0;
      a.push({
        type: se.literal,
        value: n.getDateTimeFormat(t, _).format(h)
      });
      continue;
    }
    if (Ml(f)) {
      var _ = typeof f.style == "string" ? r.time[f.style] : Sr(f.style) ? f.style.parsedOptions : r.time.medium;
      a.push({
        type: se.literal,
        value: n.getDateTimeFormat(t, _).format(h)
      });
      continue;
    }
    if (Ll(f)) {
      var _ = typeof f.style == "string" ? r.number[f.style] : Fl(f.style) ? f.style.parsedOptions : void 0;
      _ && _.scale && (h = h * (_.scale || 1)), a.push({
        type: se.literal,
        value: n.getNumberFormat(t, _).format(h)
      });
      continue;
    }
    if (Ul(f)) {
      var d = f.children, g = f.value, S = i[g];
      if (!ec(S))
        throw new Qf(g, "function", s);
      var b = fn(d, t, n, r, i, l), T = S(b.map(function(v) {
        return v.value;
      }));
      Array.isArray(T) || (T = [T]), a.push.apply(a, T.map(function(v) {
        return {
          type: typeof v == "string" ? se.literal : se.object,
          value: v
        };
      }));
    }
    if (Rl(f)) {
      var y = f.options[h] || f.options.other;
      if (!y)
        throw new Si(f.value, h, Object.keys(f.options), s);
      a.push.apply(a, fn(y.value, t, n, r, i));
      continue;
    }
    if (Dl(f)) {
      var y = f.options["=".concat(h)];
      if (!y) {
        if (!Intl.PluralRules)
          throw new yn(`Intl.PluralRules is not available in this environment.
Try polyfilling it using "@formatjs/intl-pluralrules"
`, wt.MISSING_INTL_API, s);
        var m = n.getPluralRules(t, { type: f.pluralType }).select(h - (f.offset || 0));
        y = f.options[m] || f.options.other;
      }
      if (!y)
        throw new Si(f.value, h, Object.keys(f.options), s);
      a.push.apply(a, fn(y.value, t, n, r, i, h - (f.offset || 0)));
      continue;
    }
  }
  return $f(a);
}
function tc(e, t) {
  return t ? U(U(U({}, e || {}), t || {}), Object.keys(e).reduce(function(n, r) {
    return n[r] = U(U({}, e[r]), t[r] || {}), n;
  }, {})) : e;
}
function nc(e, t) {
  return t ? Object.keys(e).reduce(function(n, r) {
    return n[r] = tc(e[r], t[r]), n;
  }, U({}, e)) : e;
}
function Yn(e) {
  return {
    create: function() {
      return {
        get: function(t) {
          return e[t];
        },
        set: function(t, n) {
          e[t] = n;
        }
      };
    }
  };
}
function rc(e) {
  return e === void 0 && (e = {
    number: {},
    dateTime: {},
    pluralRules: {}
  }), {
    getNumberFormat: Wn(function() {
      for (var t, n = [], r = 0; r < arguments.length; r++)
        n[r] = arguments[r];
      return new ((t = Intl.NumberFormat).bind.apply(t, Xn([void 0], n, !1)))();
    }, {
      cache: Yn(e.number),
      strategy: Jn.variadic
    }),
    getDateTimeFormat: Wn(function() {
      for (var t, n = [], r = 0; r < arguments.length; r++)
        n[r] = arguments[r];
      return new ((t = Intl.DateTimeFormat).bind.apply(t, Xn([void 0], n, !1)))();
    }, {
      cache: Yn(e.dateTime),
      strategy: Jn.variadic
    }),
    getPluralRules: Wn(function() {
      for (var t, n = [], r = 0; r < arguments.length; r++)
        n[r] = arguments[r];
      return new ((t = Intl.PluralRules).bind.apply(t, Xn([void 0], n, !1)))();
    }, {
      cache: Yn(e.pluralRules),
      strategy: Jn.variadic
    })
  };
}
var ic = (
  /** @class */
  function() {
    function e(t, n, r, i) {
      var l = this;
      if (n === void 0 && (n = e.defaultLocale), this.formatterCache = {
        number: {},
        dateTime: {},
        pluralRules: {}
      }, this.format = function(s) {
        var a = l.formatToParts(s);
        if (a.length === 1)
          return a[0].value;
        var o = a.reduce(function(u, f) {
          return !u.length || f.type !== se.literal || typeof u[u.length - 1] != "string" ? u.push(f.value) : u[u.length - 1] += f.value, u;
        }, []);
        return o.length <= 1 ? o[0] || "" : o;
      }, this.formatToParts = function(s) {
        return fn(l.ast, l.locales, l.formatters, l.formats, s, void 0, l.message);
      }, this.resolvedOptions = function() {
        return {
          locale: l.resolvedLocale.toString()
        };
      }, this.getAst = function() {
        return l.ast;
      }, this.locales = n, this.resolvedLocale = e.resolveLocale(n), typeof t == "string") {
        if (this.message = t, !e.__parse)
          throw new TypeError("IntlMessageFormat.__parse must be set to process `message` of type `string`");
        this.ast = e.__parse(t, {
          ignoreTag: i == null ? void 0 : i.ignoreTag,
          locale: this.resolvedLocale
        });
      } else
        this.ast = t;
      if (!Array.isArray(this.ast))
        throw new TypeError("A message must be provided as a String or AST.");
      this.formats = nc(e.formats, r), this.formatters = i && i.formatters || rc(this.formatterCache);
    }
    return Object.defineProperty(e, "defaultLocale", {
      get: function() {
        return e.memoizedDefaultLocale || (e.memoizedDefaultLocale = new Intl.NumberFormat().resolvedOptions().locale), e.memoizedDefaultLocale;
      },
      enumerable: !1,
      configurable: !0
    }), e.memoizedDefaultLocale = null, e.resolveLocale = function(t) {
      var n = Intl.NumberFormat.supportedLocalesOf(t);
      return n.length > 0 ? new Intl.Locale(n[0]) : new Intl.Locale(typeof t == "string" ? t : t[0]);
    }, e.__parse = zf, e.formats = {
      number: {
        integer: {
          maximumFractionDigits: 0
        },
        currency: {
          style: "currency"
        },
        percent: {
          style: "percent"
        }
      },
      date: {
        short: {
          month: "numeric",
          day: "numeric",
          year: "2-digit"
        },
        medium: {
          month: "short",
          day: "numeric",
          year: "numeric"
        },
        long: {
          month: "long",
          day: "numeric",
          year: "numeric"
        },
        full: {
          weekday: "long",
          month: "long",
          day: "numeric",
          year: "numeric"
        }
      },
      time: {
        short: {
          hour: "numeric",
          minute: "numeric"
        },
        medium: {
          hour: "numeric",
          minute: "numeric",
          second: "numeric"
        },
        long: {
          hour: "numeric",
          minute: "numeric",
          second: "numeric",
          timeZoneName: "short"
        },
        full: {
          hour: "numeric",
          minute: "numeric",
          second: "numeric",
          timeZoneName: "short"
        }
      }
    }, e;
  }()
);
function lc(e, t) {
  if (t == null)
    return;
  if (t in e)
    return e[t];
  const n = t.split(".");
  let r = e;
  for (let i = 0; i < n.length; i++)
    if (typeof r == "object") {
      if (i > 0) {
        const l = n.slice(i, n.length).join(".");
        if (l in r) {
          r = r[l];
          break;
        }
      }
      r = r[n[i]];
    } else
      r = void 0;
  return r;
}
const We = {}, sc = (e, t, n) => n && (t in We || (We[t] = {}), e in We[t] || (We[t][e] = n), n), Jl = (e, t) => {
  if (t == null)
    return;
  if (t in We && e in We[t])
    return We[t][e];
  const n = En(t);
  for (let r = 0; r < n.length; r++) {
    const i = n[r], l = ac(i, e);
    if (l)
      return sc(e, t, l);
  }
};
let Dr;
const Vt = zt({});
function oc(e) {
  return Dr[e] || null;
}
function Yl(e) {
  return e in Dr;
}
function ac(e, t) {
  if (!Yl(e))
    return null;
  const n = oc(e);
  return lc(n, t);
}
function uc(e) {
  if (e == null)
    return;
  const t = En(e);
  for (let n = 0; n < t.length; n++) {
    const r = t[n];
    if (Yl(r))
      return r;
  }
}
function fc(e, ...t) {
  delete We[e], Vt.update((n) => (n[e] = df.all([n[e] || {}, ...t]), n));
}
Ct(
  [Vt],
  ([e]) => Object.keys(e)
);
Vt.subscribe((e) => Dr = e);
const cn = {};
function cc(e, t) {
  cn[e].delete(t), cn[e].size === 0 && delete cn[e];
}
function Ql(e) {
  return cn[e];
}
function hc(e) {
  return En(e).map((t) => {
    const n = Ql(t);
    return [t, n ? [...n] : []];
  }).filter(([, t]) => t.length > 0);
}
function Hr(e) {
  return e == null ? !1 : En(e).some(
    (t) => {
      var n;
      return (n = Ql(t)) == null ? void 0 : n.size;
    }
  );
}
function _c(e, t) {
  return Promise.all(
    t.map((r) => (cc(e, r), r().then((i) => i.default || i)))
  ).then((r) => fc(e, ...r));
}
const Lt = {};
function Kl(e) {
  if (!Hr(e))
    return e in Lt ? Lt[e] : Promise.resolve();
  const t = hc(e);
  return Lt[e] = Promise.all(
    t.map(
      ([n, r]) => _c(n, r)
    )
  ).then(() => {
    if (Hr(e))
      return Kl(e);
    delete Lt[e];
  }), Lt[e];
}
const dc = {
  number: {
    scientific: { notation: "scientific" },
    engineering: { notation: "engineering" },
    compactLong: { notation: "compact", compactDisplay: "long" },
    compactShort: { notation: "compact", compactDisplay: "short" }
  },
  date: {
    short: { month: "numeric", day: "numeric", year: "2-digit" },
    medium: { month: "short", day: "numeric", year: "numeric" },
    long: { month: "long", day: "numeric", year: "numeric" },
    full: { weekday: "long", month: "long", day: "numeric", year: "numeric" }
  },
  time: {
    short: { hour: "numeric", minute: "numeric" },
    medium: { hour: "numeric", minute: "numeric", second: "numeric" },
    long: {
      hour: "numeric",
      minute: "numeric",
      second: "numeric",
      timeZoneName: "short"
    },
    full: {
      hour: "numeric",
      minute: "numeric",
      second: "numeric",
      timeZoneName: "short"
    }
  }
}, mc = {
  fallbackLocale: null,
  loadingDelay: 200,
  formats: dc,
  warnOnMissingMessages: !0,
  handleMissingMessage: void 0,
  ignoreTag: !0
}, gc = mc;
function vt() {
  return gc;
}
const Qn = zt(!1);
var pc = Object.defineProperty, bc = Object.defineProperties, wc = Object.getOwnPropertyDescriptors, ki = Object.getOwnPropertySymbols, vc = Object.prototype.hasOwnProperty, yc = Object.prototype.propertyIsEnumerable, Ti = (e, t, n) => t in e ? pc(e, t, { enumerable: !0, configurable: !0, writable: !0, value: n }) : e[t] = n, Ec = (e, t) => {
  for (var n in t || (t = {}))
    vc.call(t, n) && Ti(e, n, t[n]);
  if (ki)
    for (var n of ki(t))
      yc.call(t, n) && Ti(e, n, t[n]);
  return e;
}, Sc = (e, t) => bc(e, wc(t));
let Pr;
const hn = zt(null);
function Bi(e) {
  return e.split("-").map((t, n, r) => r.slice(0, n + 1).join("-")).reverse();
}
function En(e, t = vt().fallbackLocale) {
  const n = Bi(e);
  return t ? [.../* @__PURE__ */ new Set([...n, ...Bi(t)])] : n;
}
function it() {
  return Pr ?? void 0;
}
hn.subscribe((e) => {
  Pr = e ?? void 0, typeof window < "u" && e != null && document.documentElement.setAttribute("lang", e);
});
const kc = (e) => {
  if (e && uc(e) && Hr(e)) {
    const { loadingDelay: t } = vt();
    let n;
    return typeof window < "u" && it() != null && t ? n = window.setTimeout(
      () => Qn.set(!0),
      t
    ) : Qn.set(!0), Kl(e).then(() => {
      hn.set(e);
    }).finally(() => {
      clearTimeout(n), Qn.set(!1);
    });
  }
  return hn.set(e);
}, Xt = Sc(Ec({}, hn), {
  set: kc
}), Sn = (e) => {
  const t = /* @__PURE__ */ Object.create(null);
  return (r) => {
    const i = JSON.stringify(r);
    return i in t ? t[i] : t[i] = e(r);
  };
};
var Tc = Object.defineProperty, _n = Object.getOwnPropertySymbols, $l = Object.prototype.hasOwnProperty, es = Object.prototype.propertyIsEnumerable, Ai = (e, t, n) => t in e ? Tc(e, t, { enumerable: !0, configurable: !0, writable: !0, value: n }) : e[t] = n, Ur = (e, t) => {
  for (var n in t || (t = {}))
    $l.call(t, n) && Ai(e, n, t[n]);
  if (_n)
    for (var n of _n(t))
      es.call(t, n) && Ai(e, n, t[n]);
  return e;
}, Ht = (e, t) => {
  var n = {};
  for (var r in e)
    $l.call(e, r) && t.indexOf(r) < 0 && (n[r] = e[r]);
  if (e != null && _n)
    for (var r of _n(e))
      t.indexOf(r) < 0 && es.call(e, r) && (n[r] = e[r]);
  return n;
};
const Gt = (e, t) => {
  const { formats: n } = vt();
  if (e in n && t in n[e])
    return n[e][t];
  throw new Error(`[svelte-i18n] Unknown "${t}" ${e} format.`);
}, Bc = Sn(
  (e) => {
    var t = e, { locale: n, format: r } = t, i = Ht(t, ["locale", "format"]);
    if (n == null)
      throw new Error('[svelte-i18n] A "locale" must be set to format numbers');
    return r && (i = Gt("number", r)), new Intl.NumberFormat(n, i);
  }
), Ac = Sn(
  (e) => {
    var t = e, { locale: n, format: r } = t, i = Ht(t, ["locale", "format"]);
    if (n == null)
      throw new Error('[svelte-i18n] A "locale" must be set to format dates');
    return r ? i = Gt("date", r) : Object.keys(i).length === 0 && (i = Gt("date", "short")), new Intl.DateTimeFormat(n, i);
  }
), Cc = Sn(
  (e) => {
    var t = e, { locale: n, format: r } = t, i = Ht(t, ["locale", "format"]);
    if (n == null)
      throw new Error(
        '[svelte-i18n] A "locale" must be set to format time values'
      );
    return r ? i = Gt("time", r) : Object.keys(i).length === 0 && (i = Gt("time", "short")), new Intl.DateTimeFormat(n, i);
  }
), Hc = (e = {}) => {
  var t = e, {
    locale: n = it()
  } = t, r = Ht(t, [
    "locale"
  ]);
  return Bc(Ur({ locale: n }, r));
}, Pc = (e = {}) => {
  var t = e, {
    locale: n = it()
  } = t, r = Ht(t, [
    "locale"
  ]);
  return Ac(Ur({ locale: n }, r));
}, Nc = (e = {}) => {
  var t = e, {
    locale: n = it()
  } = t, r = Ht(t, [
    "locale"
  ]);
  return Cc(Ur({ locale: n }, r));
}, Ic = Sn(
  // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
  (e, t = it()) => new ic(e, t, vt().formats, {
    ignoreTag: vt().ignoreTag
  })
), Lc = (e, t = {}) => {
  var n, r, i, l;
  let s = t;
  typeof e == "object" && (s = e, e = s.id);
  const {
    values: a,
    locale: o = it(),
    default: u
  } = s;
  if (o == null)
    throw new Error(
      "[svelte-i18n] Cannot format a message without first setting the initial locale."
    );
  let f = Jl(e, o);
  if (!f)
    f = (l = (i = (r = (n = vt()).handleMissingMessage) == null ? void 0 : r.call(n, { locale: o, id: e, defaultValue: u })) != null ? i : u) != null ? l : e;
  else if (typeof f != "string")
    return console.warn(
      `[svelte-i18n] Message with id "${e}" must be of type "string", found: "${typeof f}". Gettin its value through the "$format" method is deprecated; use the "json" method instead.`
    ), f;
  if (!a)
    return f;
  let c = f;
  try {
    c = Ic(f, o).format(a);
  } catch (h) {
    h instanceof Error && console.warn(
      `[svelte-i18n] Message "${e}" has syntax error:`,
      h.message
    );
  }
  return c;
}, Oc = (e, t) => Nc(t).format(e), Mc = (e, t) => Pc(t).format(e), Rc = (e, t) => Hc(t).format(e), Dc = (e, t = it()) => Jl(e, t);
Ct([Xt, Vt], () => Lc);
Ct([Xt], () => Oc);
Ct([Xt], () => Mc);
Ct([Xt], () => Rc);
Ct([Xt, Vt], () => Dc);
const {
  SvelteComponent: Uc,
  append: ae,
  attr: $e,
  detach: ts,
  element: et,
  init: Fc,
  insert: ns,
  noop: Ci,
  safe_not_equal: Gc,
  set_data: dn,
  set_style: Kn,
  space: Nr,
  text: ut,
  toggle_class: Hi
} = window.__gradio__svelte__internal, { onMount: qc, createEventDispatcher: xc } = window.__gradio__svelte__internal;
function Pi(e) {
  let t, n, r, i, l = Mt(
    /*current_file_upload*/
    e[2]
  ) + "", s, a, o, u, f = (
    /*current_file_upload*/
    e[2].orig_name + ""
  ), c;
  return {
    c() {
      t = et("div"), n = et("span"), r = et("div"), i = et("progress"), s = ut(l), o = Nr(), u = et("span"), c = ut(f), Kn(i, "visibility", "hidden"), Kn(i, "height", "0"), Kn(i, "width", "0"), i.value = a = Mt(
        /*current_file_upload*/
        e[2]
      ), $e(i, "max", "100"), $e(i, "class", "svelte-12ckl9l"), $e(r, "class", "progress-bar svelte-12ckl9l"), $e(u, "class", "file-name svelte-12ckl9l"), $e(t, "class", "file svelte-12ckl9l");
    },
    m(h, _) {
      ns(h, t, _), ae(t, n), ae(n, r), ae(r, i), ae(i, s), ae(t, o), ae(t, u), ae(u, c);
    },
    p(h, _) {
      _ & /*current_file_upload*/
      4 && l !== (l = Mt(
        /*current_file_upload*/
        h[2]
      ) + "") && dn(s, l), _ & /*current_file_upload*/
      4 && a !== (a = Mt(
        /*current_file_upload*/
        h[2]
      )) && (i.value = a), _ & /*current_file_upload*/
      4 && f !== (f = /*current_file_upload*/
      h[2].orig_name + "") && dn(c, f);
    },
    d(h) {
      h && ts(t);
    }
  };
}
function jc(e) {
  let t, n, r, i = (
    /*files_with_progress*/
    e[0].length + ""
  ), l, s, a = (
    /*files_with_progress*/
    e[0].length > 1 ? "files" : "file"
  ), o, u, f, c = (
    /*current_file_upload*/
    e[2] && Pi(e)
  );
  return {
    c() {
      t = et("div"), n = et("span"), r = ut("Uploading "), l = ut(i), s = Nr(), o = ut(a), u = ut("..."), f = Nr(), c && c.c(), $e(n, "class", "uploading svelte-12ckl9l"), $e(t, "class", "wrap svelte-12ckl9l"), Hi(
        t,
        "progress",
        /*progress*/
        e[1]
      );
    },
    m(h, _) {
      ns(h, t, _), ae(t, n), ae(n, r), ae(n, l), ae(n, s), ae(n, o), ae(n, u), ae(t, f), c && c.m(t, null);
    },
    p(h, [_]) {
      _ & /*files_with_progress*/
      1 && i !== (i = /*files_with_progress*/
      h[0].length + "") && dn(l, i), _ & /*files_with_progress*/
      1 && a !== (a = /*files_with_progress*/
      h[0].length > 1 ? "files" : "file") && dn(o, a), /*current_file_upload*/
      h[2] ? c ? c.p(h, _) : (c = Pi(h), c.c(), c.m(t, null)) : c && (c.d(1), c = null), _ & /*progress*/
      2 && Hi(
        t,
        "progress",
        /*progress*/
        h[1]
      );
    },
    i: Ci,
    o: Ci,
    d(h) {
      h && ts(t), c && c.d();
    }
  };
}
function Mt(e) {
  return e.progress * 100 / (e.size || 0) || 0;
}
function zc(e) {
  let t = 0;
  return e.forEach((n) => {
    t += Mt(n);
  }), document.documentElement.style.setProperty("--upload-progress-width", (t / e.length).toFixed(2) + "%"), t / e.length;
}
function Vc(e, t, n) {
  let { upload_id: r } = t, { root: i } = t, { files: l } = t, s, a = !1, o, u = l.map((h) => ({ ...h, progress: 0 }));
  const f = xc();
  function c(h, _) {
    n(0, u = u.map((d) => (d.orig_name === h && (d.progress += _), d)));
  }
  return qc(() => {
    s = new EventSource(`${i}/upload_progress?upload_id=${r}`), s.onmessage = async function(h) {
      const _ = JSON.parse(h.data);
      a || n(1, a = !0), _.msg === "done" ? (s.close(), f("done")) : (n(2, o = _), c(_.orig_name, _.chunk_size));
    };
  }), e.$$set = (h) => {
    "upload_id" in h && n(3, r = h.upload_id), "root" in h && n(4, i = h.root), "files" in h && n(5, l = h.files);
  }, e.$$.update = () => {
    e.$$.dirty & /*files_with_progress*/
    1 && zc(u);
  }, [u, a, o, r, i, l];
}
class Xc extends Uc {
  constructor(t) {
    super(), Fc(this, t, Vc, jc, Gc, { upload_id: 3, root: 4, files: 5 });
  }
}
const {
  SvelteComponent: Zc,
  append: Ni,
  attr: be,
  binding_callbacks: Wc,
  bubble: Ye,
  check_outros: Jc,
  create_component: Yc,
  create_slot: Qc,
  destroy_component: Kc,
  detach: rs,
  element: Ii,
  empty: $c,
  get_all_dirty_from_scope: e0,
  get_slot_changes: t0,
  group_outros: n0,
  init: r0,
  insert: is,
  listen: we,
  mount_component: i0,
  prevent_default: Qe,
  run_all: l0,
  safe_not_equal: s0,
  set_style: Li,
  space: o0,
  stop_propagation: Ke,
  toggle_class: Ve,
  transition_in: mn,
  transition_out: gn,
  update_slot_base: a0
} = window.__gradio__svelte__internal, { createEventDispatcher: u0, tick: f0, getContext: c0 } = window.__gradio__svelte__internal;
function h0(e) {
  let t, n, r, i, l, s, a, o, u, f;
  const c = (
    /*#slots*/
    e[21].default
  ), h = Qc(
    c,
    e,
    /*$$scope*/
    e[20],
    null
  );
  return {
    c() {
      t = Ii("button"), h && h.c(), n = o0(), r = Ii("input"), be(r, "aria-label", "file upload"), be(r, "type", "file"), be(
        r,
        "accept",
        /*filetype*/
        e[1]
      ), r.multiple = i = /*file_count*/
      e[5] === "multiple" || void 0, be(r, "webkitdirectory", l = /*file_count*/
      e[5] === "directory" || void 0), be(r, "mozdirectory", s = /*file_count*/
      e[5] === "directory" || void 0), be(r, "class", "svelte-1aq8tno"), be(t, "tabindex", a = /*hidden*/
      e[7] ? -1 : 0), be(t, "class", "svelte-1aq8tno"), Ve(
        t,
        "hidden",
        /*hidden*/
        e[7]
      ), Ve(
        t,
        "center",
        /*center*/
        e[3]
      ), Ve(
        t,
        "boundedheight",
        /*boundedheight*/
        e[2]
      ), Ve(
        t,
        "flex",
        /*flex*/
        e[4]
      ), Li(
        t,
        "height",
        /*include_sources*/
        e[8] ? "calc(100% - 40px" : "100%"
      );
    },
    m(_, d) {
      is(_, t, d), h && h.m(t, null), Ni(t, n), Ni(t, r), e[29](r), o = !0, u || (f = [
        we(
          r,
          "change",
          /*load_files_from_upload*/
          e[14]
        ),
        we(t, "drag", Ke(Qe(
          /*drag_handler*/
          e[22]
        ))),
        we(t, "dragstart", Ke(Qe(
          /*dragstart_handler*/
          e[23]
        ))),
        we(t, "dragend", Ke(Qe(
          /*dragend_handler*/
          e[24]
        ))),
        we(t, "dragover", Ke(Qe(
          /*dragover_handler*/
          e[25]
        ))),
        we(t, "dragenter", Ke(Qe(
          /*dragenter_handler*/
          e[26]
        ))),
        we(t, "dragleave", Ke(Qe(
          /*dragleave_handler*/
          e[27]
        ))),
        we(t, "drop", Ke(Qe(
          /*drop_handler*/
          e[28]
        ))),
        we(
          t,
          "click",
          /*open_file_upload*/
          e[9]
        ),
        we(
          t,
          "drop",
          /*loadFilesFromDrop*/
          e[15]
        ),
        we(
          t,
          "dragenter",
          /*updateDragging*/
          e[13]
        ),
        we(
          t,
          "dragleave",
          /*updateDragging*/
          e[13]
        )
      ], u = !0);
    },
    p(_, d) {
      h && h.p && (!o || d[0] & /*$$scope*/
      1048576) && a0(
        h,
        c,
        _,
        /*$$scope*/
        _[20],
        o ? t0(
          c,
          /*$$scope*/
          _[20],
          d,
          null
        ) : e0(
          /*$$scope*/
          _[20]
        ),
        null
      ), (!o || d[0] & /*filetype*/
      2) && be(
        r,
        "accept",
        /*filetype*/
        _[1]
      ), (!o || d[0] & /*file_count*/
      32 && i !== (i = /*file_count*/
      _[5] === "multiple" || void 0)) && (r.multiple = i), (!o || d[0] & /*file_count*/
      32 && l !== (l = /*file_count*/
      _[5] === "directory" || void 0)) && be(r, "webkitdirectory", l), (!o || d[0] & /*file_count*/
      32 && s !== (s = /*file_count*/
      _[5] === "directory" || void 0)) && be(r, "mozdirectory", s), (!o || d[0] & /*hidden*/
      128 && a !== (a = /*hidden*/
      _[7] ? -1 : 0)) && be(t, "tabindex", a), (!o || d[0] & /*hidden*/
      128) && Ve(
        t,
        "hidden",
        /*hidden*/
        _[7]
      ), (!o || d[0] & /*center*/
      8) && Ve(
        t,
        "center",
        /*center*/
        _[3]
      ), (!o || d[0] & /*boundedheight*/
      4) && Ve(
        t,
        "boundedheight",
        /*boundedheight*/
        _[2]
      ), (!o || d[0] & /*flex*/
      16) && Ve(
        t,
        "flex",
        /*flex*/
        _[4]
      ), d[0] & /*include_sources*/
      256 && Li(
        t,
        "height",
        /*include_sources*/
        _[8] ? "calc(100% - 40px" : "100%"
      );
    },
    i(_) {
      o || (mn(h, _), o = !0);
    },
    o(_) {
      gn(h, _), o = !1;
    },
    d(_) {
      _ && rs(t), h && h.d(_), e[29](null), u = !1, l0(f);
    }
  };
}
function _0(e) {
  let t, n;
  return t = new Xc({
    props: {
      root: (
        /*root*/
        e[6]
      ),
      upload_id: (
        /*upload_id*/
        e[10]
      ),
      files: (
        /*file_data*/
        e[11]
      )
    }
  }), {
    c() {
      Yc(t.$$.fragment);
    },
    m(r, i) {
      i0(t, r, i), n = !0;
    },
    p(r, i) {
      const l = {};
      i[0] & /*root*/
      64 && (l.root = /*root*/
      r[6]), i[0] & /*upload_id*/
      1024 && (l.upload_id = /*upload_id*/
      r[10]), i[0] & /*file_data*/
      2048 && (l.files = /*file_data*/
      r[11]), t.$set(l);
    },
    i(r) {
      n || (mn(t.$$.fragment, r), n = !0);
    },
    o(r) {
      gn(t.$$.fragment, r), n = !1;
    },
    d(r) {
      Kc(t, r);
    }
  };
}
function d0(e) {
  let t, n, r, i;
  const l = [_0, h0], s = [];
  function a(o, u) {
    return (
      /*uploading*/
      o[0] ? 0 : 1
    );
  }
  return t = a(e), n = s[t] = l[t](e), {
    c() {
      n.c(), r = $c();
    },
    m(o, u) {
      s[t].m(o, u), is(o, r, u), i = !0;
    },
    p(o, u) {
      let f = t;
      t = a(o), t === f ? s[t].p(o, u) : (n0(), gn(s[f], 1, 1, () => {
        s[f] = null;
      }), Jc(), n = s[t], n ? n.p(o, u) : (n = s[t] = l[t](o), n.c()), mn(n, 1), n.m(r.parentNode, r));
    },
    i(o) {
      i || (mn(n), i = !0);
    },
    o(o) {
      gn(n), i = !1;
    },
    d(o) {
      o && rs(r), s[t].d(o);
    }
  };
}
function $n(e) {
  let t, n = e[0], r = 1;
  for (; r < e.length; ) {
    const i = e[r], l = e[r + 1];
    if (r += 2, (i === "optionalAccess" || i === "optionalCall") && n == null)
      return;
    i === "access" || i === "optionalAccess" ? (t = n, n = l(n)) : (i === "call" || i === "optionalCall") && (n = l((...s) => n.call(t, ...s)), t = void 0);
  }
  return n;
}
function m0(e, t) {
  return !e || e === "*" ? !0 : e.endsWith("/*") ? t.startsWith(e.slice(0, -1)) : e === t;
}
function g0(e, t, n) {
  let { $$slots: r = {}, $$scope: i } = t, { filetype: l = null } = t, { dragging: s = !1 } = t, { boundedheight: a = !0 } = t, { center: o = !0 } = t, { flex: u = !0 } = t, { file_count: f = "single" } = t, { disable_click: c = !1 } = t, { root: h } = t, { hidden: _ = !1 } = t, { format: d = "file" } = t, { include_sources: g = !1 } = t, { uploading: S = !1 } = t, b, T;
  const y = c0("upload_files");
  let m;
  const v = u0();
  function G() {
    n(16, s = !s);
  }
  function F() {
    c || (n(12, m.value = "", m), m.click());
  }
  async function q(w) {
    await f0(), n(10, b = Math.random().toString(36).substring(2, 15)), n(0, S = !0);
    const E = await Tu(w, h, b, y);
    return v("load", f === "single" ? $n([E, "optionalAccess", (A) => A[0]]) : E), n(0, S = !1), E || [];
  }
  async function K(w) {
    if (!w.length)
      return;
    let E = w.map((A) => new File([A], A.name));
    return n(11, T = await Bu(E)), await q(T);
  }
  async function M(w) {
    const E = w.target;
    if (E.files)
      if (d != "blob")
        await K(Array.from(E.files));
      else {
        if (f === "single") {
          v("load", E.files[0]);
          return;
        }
        v("load", E.files);
      }
  }
  async function ie(w) {
    if (n(16, s = !1), !$n([w, "access", (A) => A.dataTransfer, "optionalAccess", (A) => A.files]))
      return;
    const E = Array.from(w.dataTransfer.files).filter((A) => $n([
      l,
      "optionalAccess",
      (p) => p.split,
      "call",
      (p) => p(","),
      "access",
      (p) => p.some,
      "call",
      (p) => p((D) => m0(D, A.type))
    ]) ? !0 : (v("error", `Invalid file type only ${l} allowed.`), !1));
    await K(E);
  }
  function oe(w) {
    Ye.call(this, e, w);
  }
  function $(w) {
    Ye.call(this, e, w);
  }
  function H(w) {
    Ye.call(this, e, w);
  }
  function I(w) {
    Ye.call(this, e, w);
  }
  function X(w) {
    Ye.call(this, e, w);
  }
  function x(w) {
    Ye.call(this, e, w);
  }
  function C(w) {
    Ye.call(this, e, w);
  }
  function B(w) {
    Wc[w ? "unshift" : "push"](() => {
      m = w, n(12, m);
    });
  }
  return e.$$set = (w) => {
    "filetype" in w && n(1, l = w.filetype), "dragging" in w && n(16, s = w.dragging), "boundedheight" in w && n(2, a = w.boundedheight), "center" in w && n(3, o = w.center), "flex" in w && n(4, u = w.flex), "file_count" in w && n(5, f = w.file_count), "disable_click" in w && n(17, c = w.disable_click), "root" in w && n(6, h = w.root), "hidden" in w && n(7, _ = w.hidden), "format" in w && n(18, d = w.format), "include_sources" in w && n(8, g = w.include_sources), "uploading" in w && n(0, S = w.uploading), "$$scope" in w && n(20, i = w.$$scope);
  }, [
    S,
    l,
    a,
    o,
    u,
    f,
    h,
    _,
    g,
    F,
    b,
    T,
    m,
    G,
    M,
    ie,
    s,
    c,
    d,
    K,
    i,
    r,
    oe,
    $,
    H,
    I,
    X,
    x,
    C,
    B
  ];
}
class p0 extends Zc {
  constructor(t) {
    super(), r0(
      this,
      t,
      g0,
      d0,
      s0,
      {
        filetype: 1,
        dragging: 16,
        boundedheight: 2,
        center: 3,
        flex: 4,
        file_count: 5,
        disable_click: 17,
        root: 6,
        hidden: 7,
        format: 18,
        include_sources: 8,
        uploading: 0,
        open_file_upload: 9,
        load_files: 19
      },
      null,
      [-1, -1]
    );
  }
  get open_file_upload() {
    return this.$$.ctx[9];
  }
  get load_files() {
    return this.$$.ctx[19];
  }
}
const {
  SvelteComponent: b0,
  append: Oi,
  attr: w0,
  create_component: er,
  destroy_component: tr,
  detach: v0,
  element: y0,
  init: E0,
  insert: S0,
  mount_component: nr,
  noop: k0,
  safe_not_equal: T0,
  space: Mi,
  transition_in: rr,
  transition_out: ir
} = window.__gradio__svelte__internal, { createEventDispatcher: B0 } = window.__gradio__svelte__internal;
function A0(e) {
  let t, n, r, i, l, s, a;
  return n = new ht({
    props: { Icon: Aa, label: "Remove Last Box" }
  }), n.$on(
    "click",
    /*click_handler*/
    e[1]
  ), i = new ht({
    props: { Icon: sa, label: "Remove All boxes" }
  }), i.$on(
    "click",
    /*click_handler_1*/
    e[2]
  ), s = new ht({
    props: { Icon: Do, label: "Remove Image" }
  }), s.$on(
    "click",
    /*click_handler_2*/
    e[3]
  ), {
    c() {
      t = y0("div"), er(n.$$.fragment), r = Mi(), er(i.$$.fragment), l = Mi(), er(s.$$.fragment), w0(t, "class", "svelte-1o7cyxy");
    },
    m(o, u) {
      S0(o, t, u), nr(n, t, null), Oi(t, r), nr(i, t, null), Oi(t, l), nr(s, t, null), a = !0;
    },
    p: k0,
    i(o) {
      a || (rr(n.$$.fragment, o), rr(i.$$.fragment, o), rr(s.$$.fragment, o), a = !0);
    },
    o(o) {
      ir(n.$$.fragment, o), ir(i.$$.fragment, o), ir(s.$$.fragment, o), a = !1;
    },
    d(o) {
      o && v0(t), tr(n), tr(i), tr(s);
    }
  };
}
function C0(e) {
  const t = B0();
  return [t, (l) => {
    t("remove_box"), l.stopPropagation();
  }, (l) => {
    t("remove_boxes"), l.stopPropagation();
  }, (l) => {
    t("remove_image"), l.stopPropagation();
  }];
}
class H0 extends b0 {
  constructor(t) {
    super(), E0(this, t, C0, A0, T0, {});
  }
}
const {
  SvelteComponent: P0,
  append: N0,
  attr: Ri,
  binding_callbacks: Di,
  bubble: I0,
  detach: L0,
  element: Ui,
  flush: tn,
  init: O0,
  insert: M0,
  listen: Ie,
  noop: lr,
  run_all: R0,
  safe_not_equal: D0,
  set_style: U0,
  stop_propagation: F0
} = window.__gradio__svelte__internal, { createEventDispatcher: G0, onDestroy: q0, onMount: x0, tick: j0 } = window.__gradio__svelte__internal;
function z0(e) {
  let t, n, r, i;
  return {
    c() {
      t = Ui("div"), n = Ui("canvas"), U0(n, "z-index", "15"), Ri(n, "class", "svelte-1mnpmgt"), Ri(t, "class", "wrap svelte-1mnpmgt");
    },
    m(l, s) {
      M0(l, t, s), N0(t, n), e[13](n), e[14](t), r || (i = [
        Ie(
          n,
          "mousedown",
          /*handle_draw_start*/
          e[2]
        ),
        Ie(
          n,
          "mousemove",
          /*handle_draw_move*/
          e[3]
        ),
        Ie(
          n,
          "mouseout",
          /*handle_draw_move*/
          e[3]
        ),
        Ie(
          n,
          "mouseup",
          /*handle_draw_end*/
          e[4]
        ),
        Ie(
          n,
          "touchstart",
          /*handle_draw_start*/
          e[2]
        ),
        Ie(
          n,
          "touchmove",
          /*handle_draw_move*/
          e[3]
        ),
        Ie(
          n,
          "touchend",
          /*handle_draw_end*/
          e[4]
        ),
        Ie(
          n,
          "touchcancel",
          /*handle_draw_end*/
          e[4]
        ),
        Ie(
          n,
          "blur",
          /*handle_draw_end*/
          e[4]
        ),
        Ie(n, "click", F0(
          /*click_handler*/
          e[12]
        ))
      ], r = !0);
    },
    p: lr,
    i: lr,
    o: lr,
    d(l) {
      l && L0(t), e[13](null), e[14](null), r = !1, R0(i);
    }
  };
}
function V0(e, t, n) {
  const r = G0();
  let { width: i = 0 } = t, { height: l = 0 } = t, { natural_width: s = 0 } = t, { natural_height: a = 0 } = t, o = [], u = [], f, c, h, _ = !1, d, g, S, b, T, y = 0, m = 0, v;
  async function G(E) {
    await j0(), n(1, c.width = E.width, c), n(1, c.height = E.height, c), n(1, c.style.width = `${E.width}px`, c), n(1, c.style.height = `${E.height}px`, c), n(1, c.style.marginTop = `-${E.height}px`, c);
  }
  async function F() {
    i === y && l === m || (await G({ width: i, height: l }), I(), setTimeout(
      () => {
        m = l, y = i;
      },
      100
    ), q());
  }
  function q() {
    return o = [], u = [], I(), r("change", u), !0;
  }
  function K() {
    return o.pop(), u.pop(), I(), r("change", u), !0;
  }
  x0(async () => {
    h = c.getContext("2d"), h && (h.lineJoin = "round", h.lineCap = "round", h.strokeStyle = "#000"), v = new ResizeObserver(() => {
      F();
    }), v.observe(f), H(), q();
  }), q0(() => {
    v.unobserve(f);
  });
  function M(E) {
    const A = c.getBoundingClientRect();
    let p, D;
    if (E instanceof MouseEvent)
      p = E.clientX, D = E.clientY;
    else if (E instanceof TouchEvent)
      p = E.changedTouches[0].clientX, D = E.changedTouches[0].clientY;
    else
      return { x: g, y: S };
    return {
      x: p - A.left,
      y: D - A.top
    };
  }
  function ie(E) {
    E.preventDefault(), _ = !0, d = 0, E instanceof MouseEvent && (d = E.button);
    const { x: A, y: p } = M(E);
    g = A, S = p;
  }
  function oe(E) {
    E.preventDefault();
    const { x: A, y: p } = M(E);
    b = A, T = p;
  }
  function $(E) {
    if (E.preventDefault(), _) {
      const { x: A, y: p } = M(E);
      let D = Math.min(g, A), Se = Math.min(S, p), le = Math.max(g, A), ke = Math.max(S, p);
      o.push([D, Se, le, ke]);
      let _e = s / i, j = a / l, k = D == le && Se == ke;
      u.push([
        Math.round(D * _e),
        Math.round(Se * j),
        k ? d == 0 ? 1 : 0 : 2,
        k ? 0 : Math.round(
          le * _e
        ),
        k ? 0 : Math.round(ke * j),
        k ? 4 : 3
      ]), r("change", u);
    }
    _ = !1;
  }
  function H() {
    I(), window.requestAnimationFrame(() => {
      H();
    });
  }
  function I() {
    if (h)
      if (h.clearRect(0, 0, i, l), _ && b != g && S != T) {
        let E = o.slice();
        E.push([g, S, b, T]), X(E), x(o);
      } else
        X(o), x(o);
  }
  function X(E) {
    h && (h.fillStyle = "rgba(0, 0, 0, 0.1)", h.beginPath(), E.forEach((A) => {
      A[0] != A[2] && A[1] != A[3] && h.rect(A[0], A[1], A[2] - A[0], A[3] - A[1]);
    }), h.fill(), h.stroke());
  }
  function x(E) {
    h && (h.beginPath(), h.fillStyle = "rgba(0, 255, 255, 1.0)", E.forEach((A, p) => {
      if (u[p][2] == 1) {
        let D = Math.sqrt(i * l) * 0.01;
        h.moveTo(A[0] + D, A[1]), h.arc(A[0], A[1], D, 0, 2 * Math.PI, !1);
      }
    }), h.fill(), h.stroke(), h.beginPath(), h.fillStyle = "rgba(255, 192, 203, 1.0)", E.forEach((A, p) => {
      if (u[p][2] == 0) {
        let D = Math.sqrt(i * l) * 0.01;
        h.moveTo(A[0] + D, A[1]), h.arc(A[0], A[1], D, 0, 2 * Math.PI, !1);
      }
    }), h.fill(), h.stroke());
  }
  function C(E) {
    I0.call(this, e, E);
  }
  function B(E) {
    Di[E ? "unshift" : "push"](() => {
      c = E, n(1, c);
    });
  }
  function w(E) {
    Di[E ? "unshift" : "push"](() => {
      f = E, n(0, f);
    });
  }
  return e.$$set = (E) => {
    "width" in E && n(5, i = E.width), "height" in E && n(6, l = E.height), "natural_width" in E && n(7, s = E.natural_width), "natural_height" in E && n(8, a = E.natural_height);
  }, [
    f,
    c,
    ie,
    oe,
    $,
    i,
    l,
    s,
    a,
    F,
    q,
    K,
    C,
    B,
    w
  ];
}
class X0 extends P0 {
  constructor(t) {
    super(), O0(
      this,
      t,
      V0,
      z0,
      D0,
      {
        width: 5,
        height: 6,
        natural_width: 7,
        natural_height: 8,
        resize_canvas: 9,
        clear: 10,
        undo: 11
      },
      null,
      [-1, -1]
    );
  }
  get width() {
    return this.$$.ctx[5];
  }
  set width(t) {
    this.$$set({ width: t }), tn();
  }
  get height() {
    return this.$$.ctx[6];
  }
  set height(t) {
    this.$$set({ height: t }), tn();
  }
  get natural_width() {
    return this.$$.ctx[7];
  }
  set natural_width(t) {
    this.$$set({ natural_width: t }), tn();
  }
  get natural_height() {
    return this.$$.ctx[8];
  }
  set natural_height(t) {
    this.$$set({ natural_height: t }), tn();
  }
  get resize_canvas() {
    return this.$$.ctx[9];
  }
  get clear() {
    return this.$$.ctx[10];
  }
  get undo() {
    return this.$$.ctx[11];
  }
}
const {
  SvelteComponent: Z0,
  add_flush_callback: Fi,
  append: nn,
  attr: Ze,
  bind: Gi,
  binding_callbacks: pn,
  bubble: W0,
  check_outros: Dt,
  create_component: yt,
  create_slot: J0,
  destroy_component: Et,
  destroy_each: Y0,
  detach: St,
  element: Ir,
  empty: ls,
  ensure_array_like: qi,
  get_all_dirty_from_scope: Q0,
  get_slot_changes: K0,
  group_outros: Ut,
  init: $0,
  insert: kt,
  listen: xi,
  mount_component: Tt,
  noop: eh,
  run_all: th,
  safe_not_equal: nh,
  space: Rt,
  src_url_equal: ji,
  transition_in: Z,
  transition_out: ne,
  update_slot_base: rh
} = window.__gradio__svelte__internal, { createEventDispatcher: ih } = window.__gradio__svelte__internal;
function zi(e, t, n) {
  const r = e.slice();
  return r[32] = t[n], r;
}
function Vi(e) {
  let t, n;
  return t = new H0({}), t.$on(
    "remove_box",
    /*remove_box_handler*/
    e[22]
  ), t.$on(
    "remove_boxes",
    /*remove_boxes_handler*/
    e[23]
  ), t.$on(
    "remove_image",
    /*remove_image_handler*/
    e[24]
  ), {
    c() {
      yt(t.$$.fragment);
    },
    m(r, i) {
      Tt(t, r, i), n = !0;
    },
    p: eh,
    i(r) {
      n || (Z(t.$$.fragment, r), n = !0);
    },
    o(r) {
      ne(t.$$.fragment, r), n = !1;
    },
    d(r) {
      Et(t, r);
    }
  };
}
function Xi(e) {
  let t;
  const n = (
    /*#slots*/
    e[21].default
  ), r = J0(
    n,
    e,
    /*$$scope*/
    e[31],
    null
  );
  return {
    c() {
      r && r.c();
    },
    m(i, l) {
      r && r.m(i, l), t = !0;
    },
    p(i, l) {
      r && r.p && (!t || l[1] & /*$$scope*/
      1) && rh(
        r,
        n,
        i,
        /*$$scope*/
        i[31],
        t ? K0(
          n,
          /*$$scope*/
          i[31],
          l,
          null
        ) : Q0(
          /*$$scope*/
          i[31]
        ),
        null
      );
    },
    i(i) {
      t || (Z(r, i), t = !0);
    },
    o(i) {
      ne(r, i), t = !1;
    },
    d(i) {
      r && r.d(i);
    }
  };
}
function lh(e) {
  let t, n, r = (
    /*value*/
    e[0] === null && !/*active_tool*/
    e[6] && Xi(e)
  );
  return {
    c() {
      r && r.c(), t = ls();
    },
    m(i, l) {
      r && r.m(i, l), kt(i, t, l), n = !0;
    },
    p(i, l) {
      /*value*/
      i[0] === null && !/*active_tool*/
      i[6] ? r ? (r.p(i, l), l[0] & /*value, active_tool*/
      65 && Z(r, 1)) : (r = Xi(i), r.c(), Z(r, 1), r.m(t.parentNode, t)) : r && (Ut(), ne(r, 1, 1, () => {
        r = null;
      }), Dt());
    },
    i(i) {
      n || (Z(r), n = !0);
    },
    o(i) {
      ne(r), n = !1;
    },
    d(i) {
      i && St(t), r && r.d(i);
    }
  };
}
function Zi(e) {
  let t, n, r, i, l, s, a, o, u = {};
  return l = new X0({ props: u }), e[29](l), l.$on(
    "change",
    /*handle_points_change*/
    e[14]
  ), {
    c() {
      t = Ir("img"), i = Rt(), yt(l.$$.fragment), ji(t.src, n = /*value*/
      e[0].url) || Ze(t, "src", n), Ze(t, "alt", r = /*value*/
      e[0].alt_text), Ze(t, "class", "svelte-1qm7xww");
    },
    m(f, c) {
      kt(f, t, c), kt(f, i, c), Tt(l, f, c), s = !0, a || (o = [
        xi(
          t,
          "click",
          /*handle_click*/
          e[16]
        ),
        xi(
          t,
          "load",
          /*handle_image_load*/
          e[13]
        )
      ], a = !0);
    },
    p(f, c) {
      (!s || c[0] & /*value*/
      1 && !ji(t.src, n = /*value*/
      f[0].url)) && Ze(t, "src", n), (!s || c[0] & /*value*/
      1 && r !== (r = /*value*/
      f[0].alt_text)) && Ze(t, "alt", r);
      const h = {};
      l.$set(h);
    },
    i(f) {
      s || (Z(l.$$.fragment, f), s = !0);
    },
    o(f) {
      ne(l.$$.fragment, f), s = !1;
    },
    d(f) {
      f && (St(t), St(i)), e[29](null), Et(l, f), a = !1, th(o);
    }
  };
}
function Wi(e) {
  var r;
  let t, n;
  return t = new bu({
    props: {
      show_border: !/*value*/
      ((r = e[0]) != null && r.url),
      $$slots: { default: [sh] },
      $$scope: { ctx: e }
    }
  }), {
    c() {
      yt(t.$$.fragment);
    },
    m(i, l) {
      Tt(t, i, l), n = !0;
    },
    p(i, l) {
      var a;
      const s = {};
      l[0] & /*value*/
      1 && (s.show_border = !/*value*/
      ((a = i[0]) != null && a.url)), l[0] & /*sources_list*/
      2048 | l[1] & /*$$scope*/
      1 && (s.$$scope = { dirty: l, ctx: i }), t.$set(s);
    },
    i(i) {
      n || (Z(t.$$.fragment, i), n = !0);
    },
    o(i) {
      ne(t.$$.fragment, i), n = !1;
    },
    d(i) {
      Et(t, i);
    }
  };
}
function Ji(e) {
  let t, n;
  function r() {
    return (
      /*click_handler*/
      e[30](
        /*source*/
        e[32]
      )
    );
  }
  return t = new ht({
    props: {
      Icon: (
        /*sources_meta*/
        e[17][
          /*source*/
          e[32]
        ].icon
      ),
      size: "large",
      label: (
        /*source*/
        e[32] + "-image-toolbar-btn"
      ),
      padded: !1
    }
  }), t.$on("click", r), {
    c() {
      yt(t.$$.fragment);
    },
    m(i, l) {
      Tt(t, i, l), n = !0;
    },
    p(i, l) {
      e = i;
      const s = {};
      l[0] & /*sources_list*/
      2048 && (s.Icon = /*sources_meta*/
      e[17][
        /*source*/
        e[32]
      ].icon), l[0] & /*sources_list*/
      2048 && (s.label = /*source*/
      e[32] + "-image-toolbar-btn"), t.$set(s);
    },
    i(i) {
      n || (Z(t.$$.fragment, i), n = !0);
    },
    o(i) {
      ne(t.$$.fragment, i), n = !1;
    },
    d(i) {
      Et(t, i);
    }
  };
}
function sh(e) {
  let t, n, r = qi(
    /*sources_list*/
    e[11]
  ), i = [];
  for (let s = 0; s < r.length; s += 1)
    i[s] = Ji(zi(e, r, s));
  const l = (s) => ne(i[s], 1, 1, () => {
    i[s] = null;
  });
  return {
    c() {
      for (let s = 0; s < i.length; s += 1)
        i[s].c();
      t = ls();
    },
    m(s, a) {
      for (let o = 0; o < i.length; o += 1)
        i[o] && i[o].m(s, a);
      kt(s, t, a), n = !0;
    },
    p(s, a) {
      if (a[0] & /*sources_meta, sources_list, handle_toolbar*/
      395264) {
        r = qi(
          /*sources_list*/
          s[11]
        );
        let o;
        for (o = 0; o < r.length; o += 1) {
          const u = zi(s, r, o);
          i[o] ? (i[o].p(u, a), Z(i[o], 1)) : (i[o] = Ji(u), i[o].c(), Z(i[o], 1), i[o].m(t.parentNode, t));
        }
        for (Ut(), o = r.length; o < i.length; o += 1)
          l(o);
        Dt();
      }
    },
    i(s) {
      if (!n) {
        for (let a = 0; a < r.length; a += 1)
          Z(i[a]);
        n = !0;
      }
    },
    o(s) {
      i = i.filter(Boolean);
      for (let a = 0; a < i.length; a += 1)
        ne(i[a]);
      n = !1;
    },
    d(s) {
      s && St(t), Y0(i, s);
    }
  };
}
function oh(e) {
  var y;
  let t, n, r, i, l, s, a, o, u, f, c = (
    /*sources*/
    e[3].length > 1 || /*sources*/
    e[3].includes("clipboard")
  ), h;
  t = new yl({
    props: {
      show_label: (
        /*show_label*/
        e[2]
      ),
      Icon: wn,
      label: (
        /*label*/
        e[1] || "Image"
      )
    }
  });
  let _ = (
    /*value*/
    ((y = e[0]) == null ? void 0 : y.url) && Vi(e)
  );
  function d(m) {
    e[26](m);
  }
  function g(m) {
    e[27](m);
  }
  let S = {
    hidden: (
      /*value*/
      e[0] !== null || /*active_tool*/
      e[6] === "webcam"
    ),
    filetype: "image/*",
    root: (
      /*root*/
      e[5]
    ),
    disable_click: !/*sources*/
    e[3].includes("upload"),
    $$slots: { default: [lh] },
    $$scope: { ctx: e }
  };
  /*uploading*/
  e[7] !== void 0 && (S.uploading = /*uploading*/
  e[7]), /*dragging*/
  e[8] !== void 0 && (S.dragging = /*dragging*/
  e[8]), s = new p0({ props: S }), e[25](s), pn.push(() => Gi(s, "uploading", d)), pn.push(() => Gi(s, "dragging", g)), s.$on(
    "load",
    /*handle_upload*/
    e[15]
  ), s.$on(
    "error",
    /*error_handler*/
    e[28]
  );
  let b = (
    /*value*/
    e[0] !== null && !/*streaming*/
    e[4] && Zi(e)
  ), T = c && Wi(e);
  return {
    c() {
      yt(t.$$.fragment), n = Rt(), r = Ir("div"), _ && _.c(), i = Rt(), l = Ir("div"), yt(s.$$.fragment), u = Rt(), b && b.c(), f = Rt(), T && T.c(), Ze(l, "class", "upload-container svelte-1qm7xww"), Ze(r, "data-testid", "image"), Ze(r, "class", "image-container svelte-1qm7xww");
    },
    m(m, v) {
      Tt(t, m, v), kt(m, n, v), kt(m, r, v), _ && _.m(r, null), nn(r, i), nn(r, l), Tt(s, l, null), nn(l, u), b && b.m(l, null), nn(r, f), T && T.m(r, null), h = !0;
    },
    p(m, v) {
      var q;
      const G = {};
      v[0] & /*show_label*/
      4 && (G.show_label = /*show_label*/
      m[2]), v[0] & /*label*/
      2 && (G.label = /*label*/
      m[1] || "Image"), t.$set(G), /*value*/
      (q = m[0]) != null && q.url ? _ ? (_.p(m, v), v[0] & /*value*/
      1 && Z(_, 1)) : (_ = Vi(m), _.c(), Z(_, 1), _.m(r, i)) : _ && (Ut(), ne(_, 1, 1, () => {
        _ = null;
      }), Dt());
      const F = {};
      v[0] & /*value, active_tool*/
      65 && (F.hidden = /*value*/
      m[0] !== null || /*active_tool*/
      m[6] === "webcam"), v[0] & /*root*/
      32 && (F.root = /*root*/
      m[5]), v[0] & /*sources*/
      8 && (F.disable_click = !/*sources*/
      m[3].includes("upload")), v[0] & /*value, active_tool*/
      65 | v[1] & /*$$scope*/
      1 && (F.$$scope = { dirty: v, ctx: m }), !a && v[0] & /*uploading*/
      128 && (a = !0, F.uploading = /*uploading*/
      m[7], Fi(() => a = !1)), !o && v[0] & /*dragging*/
      256 && (o = !0, F.dragging = /*dragging*/
      m[8], Fi(() => o = !1)), s.$set(F), /*value*/
      m[0] !== null && !/*streaming*/
      m[4] ? b ? (b.p(m, v), v[0] & /*value, streaming*/
      17 && Z(b, 1)) : (b = Zi(m), b.c(), Z(b, 1), b.m(l, null)) : b && (Ut(), ne(b, 1, 1, () => {
        b = null;
      }), Dt()), v[0] & /*sources*/
      8 && (c = /*sources*/
      m[3].length > 1 || /*sources*/
      m[3].includes("clipboard")), c ? T ? (T.p(m, v), v[0] & /*sources*/
      8 && Z(T, 1)) : (T = Wi(m), T.c(), Z(T, 1), T.m(r, null)) : T && (Ut(), ne(T, 1, 1, () => {
        T = null;
      }), Dt());
    },
    i(m) {
      h || (Z(t.$$.fragment, m), Z(_), Z(s.$$.fragment, m), Z(b), Z(T), h = !0);
    },
    o(m) {
      ne(t.$$.fragment, m), ne(_), ne(s.$$.fragment, m), ne(b), ne(T), h = !1;
    },
    d(m) {
      m && (St(n), St(r)), Et(t, m), _ && _.d(), e[25](null), Et(s), b && b.d(), T && T.d();
    }
  };
}
function ah(e) {
  let t, n = e[0], r = 1;
  for (; r < e.length; ) {
    const i = e[r], l = e[r + 1];
    if (r += 2, (i === "optionalAccess" || i === "optionalCall") && n == null)
      return;
    i === "access" || i === "optionalAccess" ? (t = n, n = l(n)) : (i === "call" || i === "optionalCall") && (n = l((...s) => n.call(t, ...s)), t = void 0);
  }
  return n;
}
function uh(e, t, n) {
  let r, { $$slots: i = {}, $$scope: l } = t;
  const s = ih();
  let a, { value: o } = t, { points: u } = t, { label: f = void 0 } = t, { show_label: c } = t;
  function h(B) {
    const w = B.currentTarget;
    n(9, a.width = w.width, a), n(9, a.height = w.height, a), n(9, a.natural_width = w.naturalWidth, a), n(9, a.natural_height = w.naturalHeight, a), a.resize_canvas();
  }
  function _({ detail: B }) {
    n(19, u = B), s("points_change", B);
  }
  let { sources: d = ["upload", "clipboard"] } = t, { streaming: g = !1 } = t, { root: S } = t, { i18n: b } = t, T, y = !1, { active_tool: m = null } = t;
  function v({ detail: B }) {
    n(0, o = Re(B, S, null)), s("upload", B);
  }
  let G = !1;
  function F(B) {
    let w = Bl(B);
    w && s("select", { index: w, value: null });
  }
  const q = {
    upload: {
      icon: Tl,
      label: b("Upload"),
      order: 0
    },
    clipboard: {
      icon: va,
      label: b("Paste"),
      order: 2
    }
  };
  async function K(B) {
    switch (B) {
      case "clipboard":
        navigator.clipboard.read().then(async (w) => {
          for (let E = 0; E < w.length; E++) {
            const A = w[E].types.find((p) => p.startsWith("image/"));
            if (A) {
              n(0, o = null), w[E].getType(A).then(async (p) => {
                const D = await T.load_files([new File([p], `clipboard.${A.replace("image/", "")}`)]);
                n(0, o = ah([D, "optionalAccess", (Se) => Se[0]]) || null);
              });
              break;
            }
          }
        });
        break;
      case "upload":
        T.open_file_upload();
        break;
    }
  }
  const M = () => {
    a.undo();
  }, ie = () => {
    a.clear();
  }, oe = () => {
    n(0, o = null), s("clear");
  };
  function $(B) {
    pn[B ? "unshift" : "push"](() => {
      T = B, n(10, T);
    });
  }
  function H(B) {
    y = B, n(7, y);
  }
  function I(B) {
    G = B, n(8, G);
  }
  function X(B) {
    W0.call(this, e, B);
  }
  function x(B) {
    pn[B ? "unshift" : "push"](() => {
      a = B, n(9, a);
    });
  }
  const C = (B) => K(B);
  return e.$$set = (B) => {
    "value" in B && n(0, o = B.value), "points" in B && n(19, u = B.points), "label" in B && n(1, f = B.label), "show_label" in B && n(2, c = B.show_label), "sources" in B && n(3, d = B.sources), "streaming" in B && n(4, g = B.streaming), "root" in B && n(5, S = B.root), "i18n" in B && n(20, b = B.i18n), "active_tool" in B && n(6, m = B.active_tool), "$$scope" in B && n(31, l = B.$$scope);
  }, e.$$.update = () => {
    e.$$.dirty[0] & /*uploading*/
    128 && y && n(0, o = null), e.$$.dirty[0] & /*value, root*/
    33 && o && !o.url && n(0, o = Re(o, S, null)), e.$$.dirty[0] & /*dragging*/
    256 && s("drag", G), e.$$.dirty[0] & /*sources*/
    8 && n(11, r = d.sort((B, w) => q[B].order - q[w].order));
  }, [
    o,
    f,
    c,
    d,
    g,
    S,
    m,
    y,
    G,
    a,
    T,
    r,
    s,
    h,
    _,
    v,
    F,
    q,
    K,
    u,
    b,
    i,
    M,
    ie,
    oe,
    $,
    H,
    I,
    X,
    x,
    C,
    l
  ];
}
class fh extends Z0 {
  constructor(t) {
    super(), $0(
      this,
      t,
      uh,
      oh,
      nh,
      {
        value: 0,
        points: 19,
        label: 1,
        show_label: 2,
        sources: 3,
        streaming: 4,
        root: 5,
        i18n: 20,
        active_tool: 6
      },
      null,
      [-1, -1]
    );
  }
}
function ft(e) {
  let t = ["", "k", "M", "G", "T", "P", "E", "Z"], n = 0;
  for (; e > 1e3 && n < t.length - 1; )
    e /= 1e3, n++;
  let r = t[n];
  return (Number.isInteger(e) ? e : e.toFixed(1)) + r;
}
const {
  SvelteComponent: ch,
  append: Ce,
  attr: R,
  component_subscribe: Yi,
  detach: hh,
  element: _h,
  init: dh,
  insert: mh,
  noop: Qi,
  safe_not_equal: gh,
  set_style: rn,
  svg_element: He,
  toggle_class: Ki
} = window.__gradio__svelte__internal, { onMount: ph } = window.__gradio__svelte__internal;
function bh(e) {
  let t, n, r, i, l, s, a, o, u, f, c, h;
  return {
    c() {
      t = _h("div"), n = He("svg"), r = He("g"), i = He("path"), l = He("path"), s = He("path"), a = He("path"), o = He("g"), u = He("path"), f = He("path"), c = He("path"), h = He("path"), R(i, "d", "M255.926 0.754768L509.702 139.936V221.027L255.926 81.8465V0.754768Z"), R(i, "fill", "#FF7C00"), R(i, "fill-opacity", "0.4"), R(i, "class", "svelte-43sxxs"), R(l, "d", "M509.69 139.936L254.981 279.641V361.255L509.69 221.55V139.936Z"), R(l, "fill", "#FF7C00"), R(l, "class", "svelte-43sxxs"), R(s, "d", "M0.250138 139.937L254.981 279.641V361.255L0.250138 221.55V139.937Z"), R(s, "fill", "#FF7C00"), R(s, "fill-opacity", "0.4"), R(s, "class", "svelte-43sxxs"), R(a, "d", "M255.923 0.232622L0.236328 139.936V221.55L255.923 81.8469V0.232622Z"), R(a, "fill", "#FF7C00"), R(a, "class", "svelte-43sxxs"), rn(r, "transform", "translate(" + /*$top*/
      e[1][0] + "px, " + /*$top*/
      e[1][1] + "px)"), R(u, "d", "M255.926 141.5L509.702 280.681V361.773L255.926 222.592V141.5Z"), R(u, "fill", "#FF7C00"), R(u, "fill-opacity", "0.4"), R(u, "class", "svelte-43sxxs"), R(f, "d", "M509.69 280.679L254.981 420.384V501.998L509.69 362.293V280.679Z"), R(f, "fill", "#FF7C00"), R(f, "class", "svelte-43sxxs"), R(c, "d", "M0.250138 280.681L254.981 420.386V502L0.250138 362.295V280.681Z"), R(c, "fill", "#FF7C00"), R(c, "fill-opacity", "0.4"), R(c, "class", "svelte-43sxxs"), R(h, "d", "M255.923 140.977L0.236328 280.68V362.294L255.923 222.591V140.977Z"), R(h, "fill", "#FF7C00"), R(h, "class", "svelte-43sxxs"), rn(o, "transform", "translate(" + /*$bottom*/
      e[2][0] + "px, " + /*$bottom*/
      e[2][1] + "px)"), R(n, "viewBox", "-1200 -1200 3000 3000"), R(n, "fill", "none"), R(n, "xmlns", "http://www.w3.org/2000/svg"), R(n, "class", "svelte-43sxxs"), R(t, "class", "svelte-43sxxs"), Ki(
        t,
        "margin",
        /*margin*/
        e[0]
      );
    },
    m(_, d) {
      mh(_, t, d), Ce(t, n), Ce(n, r), Ce(r, i), Ce(r, l), Ce(r, s), Ce(r, a), Ce(n, o), Ce(o, u), Ce(o, f), Ce(o, c), Ce(o, h);
    },
    p(_, [d]) {
      d & /*$top*/
      2 && rn(r, "transform", "translate(" + /*$top*/
      _[1][0] + "px, " + /*$top*/
      _[1][1] + "px)"), d & /*$bottom*/
      4 && rn(o, "transform", "translate(" + /*$bottom*/
      _[2][0] + "px, " + /*$bottom*/
      _[2][1] + "px)"), d & /*margin*/
      1 && Ki(
        t,
        "margin",
        /*margin*/
        _[0]
      );
    },
    i: Qi,
    o: Qi,
    d(_) {
      _ && hh(t);
    }
  };
}
function wh(e, t, n) {
  let r, i, { margin: l = !0 } = t;
  const s = di([0, 0]);
  Yi(e, s, (h) => n(1, r = h));
  const a = di([0, 0]);
  Yi(e, a, (h) => n(2, i = h));
  let o;
  async function u() {
    await Promise.all([s.set([125, 140]), a.set([-125, -140])]), await Promise.all([s.set([-125, 140]), a.set([125, -140])]), await Promise.all([s.set([-125, 0]), a.set([125, -0])]), await Promise.all([s.set([125, 0]), a.set([-125, 0])]);
  }
  async function f() {
    await u(), o || f();
  }
  async function c() {
    await Promise.all([s.set([125, 0]), a.set([-125, 0])]), f();
  }
  return ph(() => (c(), () => o = !0)), e.$$set = (h) => {
    "margin" in h && n(0, l = h.margin);
  }, [l, r, i, s, a];
}
class vh extends ch {
  constructor(t) {
    super(), dh(this, t, wh, bh, gh, { margin: 0 });
  }
}
const {
  SvelteComponent: yh,
  append: nt,
  attr: Le,
  binding_callbacks: $i,
  check_outros: ss,
  create_component: Eh,
  create_slot: Sh,
  destroy_component: kh,
  destroy_each: os,
  detach: P,
  element: De,
  empty: Pt,
  ensure_array_like: bn,
  get_all_dirty_from_scope: Th,
  get_slot_changes: Bh,
  group_outros: as,
  init: Ah,
  insert: N,
  mount_component: Ch,
  noop: Lr,
  safe_not_equal: Hh,
  set_data: Ee,
  set_style: Je,
  space: Oe,
  text: V,
  toggle_class: ve,
  transition_in: Bt,
  transition_out: At,
  update_slot_base: Ph
} = window.__gradio__svelte__internal, { tick: Nh } = window.__gradio__svelte__internal, { onDestroy: Ih } = window.__gradio__svelte__internal, Lh = (e) => ({}), el = (e) => ({});
function tl(e, t, n) {
  const r = e.slice();
  return r[38] = t[n], r[40] = n, r;
}
function nl(e, t, n) {
  const r = e.slice();
  return r[38] = t[n], r;
}
function Oh(e) {
  let t, n = (
    /*i18n*/
    e[1]("common.error") + ""
  ), r, i, l;
  const s = (
    /*#slots*/
    e[29].error
  ), a = Sh(
    s,
    e,
    /*$$scope*/
    e[28],
    el
  );
  return {
    c() {
      t = De("span"), r = V(n), i = Oe(), a && a.c(), Le(t, "class", "error svelte-1txqlrd");
    },
    m(o, u) {
      N(o, t, u), nt(t, r), N(o, i, u), a && a.m(o, u), l = !0;
    },
    p(o, u) {
      (!l || u[0] & /*i18n*/
      2) && n !== (n = /*i18n*/
      o[1]("common.error") + "") && Ee(r, n), a && a.p && (!l || u[0] & /*$$scope*/
      268435456) && Ph(
        a,
        s,
        o,
        /*$$scope*/
        o[28],
        l ? Bh(
          s,
          /*$$scope*/
          o[28],
          u,
          Lh
        ) : Th(
          /*$$scope*/
          o[28]
        ),
        el
      );
    },
    i(o) {
      l || (Bt(a, o), l = !0);
    },
    o(o) {
      At(a, o), l = !1;
    },
    d(o) {
      o && (P(t), P(i)), a && a.d(o);
    }
  };
}
function Mh(e) {
  let t, n, r, i, l, s, a, o, u, f = (
    /*variant*/
    e[8] === "default" && /*show_eta_bar*/
    e[18] && /*show_progress*/
    e[6] === "full" && rl(e)
  );
  function c(y, m) {
    if (
      /*progress*/
      y[7]
    )
      return Uh;
    if (
      /*queue_position*/
      y[2] !== null && /*queue_size*/
      y[3] !== void 0 && /*queue_position*/
      y[2] >= 0
    )
      return Dh;
    if (
      /*queue_position*/
      y[2] === 0
    )
      return Rh;
  }
  let h = c(e), _ = h && h(e), d = (
    /*timer*/
    e[5] && sl(e)
  );
  const g = [xh, qh], S = [];
  function b(y, m) {
    return (
      /*last_progress_level*/
      y[15] != null ? 0 : (
        /*show_progress*/
        y[6] === "full" ? 1 : -1
      )
    );
  }
  ~(l = b(e)) && (s = S[l] = g[l](e));
  let T = !/*timer*/
  e[5] && _l(e);
  return {
    c() {
      f && f.c(), t = Oe(), n = De("div"), _ && _.c(), r = Oe(), d && d.c(), i = Oe(), s && s.c(), a = Oe(), T && T.c(), o = Pt(), Le(n, "class", "progress-text svelte-1txqlrd"), ve(
        n,
        "meta-text-center",
        /*variant*/
        e[8] === "center"
      ), ve(
        n,
        "meta-text",
        /*variant*/
        e[8] === "default"
      );
    },
    m(y, m) {
      f && f.m(y, m), N(y, t, m), N(y, n, m), _ && _.m(n, null), nt(n, r), d && d.m(n, null), N(y, i, m), ~l && S[l].m(y, m), N(y, a, m), T && T.m(y, m), N(y, o, m), u = !0;
    },
    p(y, m) {
      /*variant*/
      y[8] === "default" && /*show_eta_bar*/
      y[18] && /*show_progress*/
      y[6] === "full" ? f ? f.p(y, m) : (f = rl(y), f.c(), f.m(t.parentNode, t)) : f && (f.d(1), f = null), h === (h = c(y)) && _ ? _.p(y, m) : (_ && _.d(1), _ = h && h(y), _ && (_.c(), _.m(n, r))), /*timer*/
      y[5] ? d ? d.p(y, m) : (d = sl(y), d.c(), d.m(n, null)) : d && (d.d(1), d = null), (!u || m[0] & /*variant*/
      256) && ve(
        n,
        "meta-text-center",
        /*variant*/
        y[8] === "center"
      ), (!u || m[0] & /*variant*/
      256) && ve(
        n,
        "meta-text",
        /*variant*/
        y[8] === "default"
      );
      let v = l;
      l = b(y), l === v ? ~l && S[l].p(y, m) : (s && (as(), At(S[v], 1, 1, () => {
        S[v] = null;
      }), ss()), ~l ? (s = S[l], s ? s.p(y, m) : (s = S[l] = g[l](y), s.c()), Bt(s, 1), s.m(a.parentNode, a)) : s = null), /*timer*/
      y[5] ? T && (T.d(1), T = null) : T ? T.p(y, m) : (T = _l(y), T.c(), T.m(o.parentNode, o));
    },
    i(y) {
      u || (Bt(s), u = !0);
    },
    o(y) {
      At(s), u = !1;
    },
    d(y) {
      y && (P(t), P(n), P(i), P(a), P(o)), f && f.d(y), _ && _.d(), d && d.d(), ~l && S[l].d(y), T && T.d(y);
    }
  };
}
function rl(e) {
  let t, n = `translateX(${/*eta_level*/
  (e[17] || 0) * 100 - 100}%)`;
  return {
    c() {
      t = De("div"), Le(t, "class", "eta-bar svelte-1txqlrd"), Je(t, "transform", n);
    },
    m(r, i) {
      N(r, t, i);
    },
    p(r, i) {
      i[0] & /*eta_level*/
      131072 && n !== (n = `translateX(${/*eta_level*/
      (r[17] || 0) * 100 - 100}%)`) && Je(t, "transform", n);
    },
    d(r) {
      r && P(t);
    }
  };
}
function Rh(e) {
  let t;
  return {
    c() {
      t = V("processing |");
    },
    m(n, r) {
      N(n, t, r);
    },
    p: Lr,
    d(n) {
      n && P(t);
    }
  };
}
function Dh(e) {
  let t, n = (
    /*queue_position*/
    e[2] + 1 + ""
  ), r, i, l, s;
  return {
    c() {
      t = V("queue: "), r = V(n), i = V("/"), l = V(
        /*queue_size*/
        e[3]
      ), s = V(" |");
    },
    m(a, o) {
      N(a, t, o), N(a, r, o), N(a, i, o), N(a, l, o), N(a, s, o);
    },
    p(a, o) {
      o[0] & /*queue_position*/
      4 && n !== (n = /*queue_position*/
      a[2] + 1 + "") && Ee(r, n), o[0] & /*queue_size*/
      8 && Ee(
        l,
        /*queue_size*/
        a[3]
      );
    },
    d(a) {
      a && (P(t), P(r), P(i), P(l), P(s));
    }
  };
}
function Uh(e) {
  let t, n = bn(
    /*progress*/
    e[7]
  ), r = [];
  for (let i = 0; i < n.length; i += 1)
    r[i] = ll(nl(e, n, i));
  return {
    c() {
      for (let i = 0; i < r.length; i += 1)
        r[i].c();
      t = Pt();
    },
    m(i, l) {
      for (let s = 0; s < r.length; s += 1)
        r[s] && r[s].m(i, l);
      N(i, t, l);
    },
    p(i, l) {
      if (l[0] & /*progress*/
      128) {
        n = bn(
          /*progress*/
          i[7]
        );
        let s;
        for (s = 0; s < n.length; s += 1) {
          const a = nl(i, n, s);
          r[s] ? r[s].p(a, l) : (r[s] = ll(a), r[s].c(), r[s].m(t.parentNode, t));
        }
        for (; s < r.length; s += 1)
          r[s].d(1);
        r.length = n.length;
      }
    },
    d(i) {
      i && P(t), os(r, i);
    }
  };
}
function il(e) {
  let t, n = (
    /*p*/
    e[38].unit + ""
  ), r, i, l = " ", s;
  function a(f, c) {
    return (
      /*p*/
      f[38].length != null ? Gh : Fh
    );
  }
  let o = a(e), u = o(e);
  return {
    c() {
      u.c(), t = Oe(), r = V(n), i = V(" | "), s = V(l);
    },
    m(f, c) {
      u.m(f, c), N(f, t, c), N(f, r, c), N(f, i, c), N(f, s, c);
    },
    p(f, c) {
      o === (o = a(f)) && u ? u.p(f, c) : (u.d(1), u = o(f), u && (u.c(), u.m(t.parentNode, t))), c[0] & /*progress*/
      128 && n !== (n = /*p*/
      f[38].unit + "") && Ee(r, n);
    },
    d(f) {
      f && (P(t), P(r), P(i), P(s)), u.d(f);
    }
  };
}
function Fh(e) {
  let t = ft(
    /*p*/
    e[38].index || 0
  ) + "", n;
  return {
    c() {
      n = V(t);
    },
    m(r, i) {
      N(r, n, i);
    },
    p(r, i) {
      i[0] & /*progress*/
      128 && t !== (t = ft(
        /*p*/
        r[38].index || 0
      ) + "") && Ee(n, t);
    },
    d(r) {
      r && P(n);
    }
  };
}
function Gh(e) {
  let t = ft(
    /*p*/
    e[38].index || 0
  ) + "", n, r, i = ft(
    /*p*/
    e[38].length
  ) + "", l;
  return {
    c() {
      n = V(t), r = V("/"), l = V(i);
    },
    m(s, a) {
      N(s, n, a), N(s, r, a), N(s, l, a);
    },
    p(s, a) {
      a[0] & /*progress*/
      128 && t !== (t = ft(
        /*p*/
        s[38].index || 0
      ) + "") && Ee(n, t), a[0] & /*progress*/
      128 && i !== (i = ft(
        /*p*/
        s[38].length
      ) + "") && Ee(l, i);
    },
    d(s) {
      s && (P(n), P(r), P(l));
    }
  };
}
function ll(e) {
  let t, n = (
    /*p*/
    e[38].index != null && il(e)
  );
  return {
    c() {
      n && n.c(), t = Pt();
    },
    m(r, i) {
      n && n.m(r, i), N(r, t, i);
    },
    p(r, i) {
      /*p*/
      r[38].index != null ? n ? n.p(r, i) : (n = il(r), n.c(), n.m(t.parentNode, t)) : n && (n.d(1), n = null);
    },
    d(r) {
      r && P(t), n && n.d(r);
    }
  };
}
function sl(e) {
  let t, n = (
    /*eta*/
    e[0] ? `/${/*formatted_eta*/
    e[19]}` : ""
  ), r, i;
  return {
    c() {
      t = V(
        /*formatted_timer*/
        e[20]
      ), r = V(n), i = V("s");
    },
    m(l, s) {
      N(l, t, s), N(l, r, s), N(l, i, s);
    },
    p(l, s) {
      s[0] & /*formatted_timer*/
      1048576 && Ee(
        t,
        /*formatted_timer*/
        l[20]
      ), s[0] & /*eta, formatted_eta*/
      524289 && n !== (n = /*eta*/
      l[0] ? `/${/*formatted_eta*/
      l[19]}` : "") && Ee(r, n);
    },
    d(l) {
      l && (P(t), P(r), P(i));
    }
  };
}
function qh(e) {
  let t, n;
  return t = new vh({
    props: { margin: (
      /*variant*/
      e[8] === "default"
    ) }
  }), {
    c() {
      Eh(t.$$.fragment);
    },
    m(r, i) {
      Ch(t, r, i), n = !0;
    },
    p(r, i) {
      const l = {};
      i[0] & /*variant*/
      256 && (l.margin = /*variant*/
      r[8] === "default"), t.$set(l);
    },
    i(r) {
      n || (Bt(t.$$.fragment, r), n = !0);
    },
    o(r) {
      At(t.$$.fragment, r), n = !1;
    },
    d(r) {
      kh(t, r);
    }
  };
}
function xh(e) {
  let t, n, r, i, l, s = `${/*last_progress_level*/
  e[15] * 100}%`, a = (
    /*progress*/
    e[7] != null && ol(e)
  );
  return {
    c() {
      t = De("div"), n = De("div"), a && a.c(), r = Oe(), i = De("div"), l = De("div"), Le(n, "class", "progress-level-inner svelte-1txqlrd"), Le(l, "class", "progress-bar svelte-1txqlrd"), Je(l, "width", s), Le(i, "class", "progress-bar-wrap svelte-1txqlrd"), Le(t, "class", "progress-level svelte-1txqlrd");
    },
    m(o, u) {
      N(o, t, u), nt(t, n), a && a.m(n, null), nt(t, r), nt(t, i), nt(i, l), e[30](l);
    },
    p(o, u) {
      /*progress*/
      o[7] != null ? a ? a.p(o, u) : (a = ol(o), a.c(), a.m(n, null)) : a && (a.d(1), a = null), u[0] & /*last_progress_level*/
      32768 && s !== (s = `${/*last_progress_level*/
      o[15] * 100}%`) && Je(l, "width", s);
    },
    i: Lr,
    o: Lr,
    d(o) {
      o && P(t), a && a.d(), e[30](null);
    }
  };
}
function ol(e) {
  let t, n = bn(
    /*progress*/
    e[7]
  ), r = [];
  for (let i = 0; i < n.length; i += 1)
    r[i] = hl(tl(e, n, i));
  return {
    c() {
      for (let i = 0; i < r.length; i += 1)
        r[i].c();
      t = Pt();
    },
    m(i, l) {
      for (let s = 0; s < r.length; s += 1)
        r[s] && r[s].m(i, l);
      N(i, t, l);
    },
    p(i, l) {
      if (l[0] & /*progress_level, progress*/
      16512) {
        n = bn(
          /*progress*/
          i[7]
        );
        let s;
        for (s = 0; s < n.length; s += 1) {
          const a = tl(i, n, s);
          r[s] ? r[s].p(a, l) : (r[s] = hl(a), r[s].c(), r[s].m(t.parentNode, t));
        }
        for (; s < r.length; s += 1)
          r[s].d(1);
        r.length = n.length;
      }
    },
    d(i) {
      i && P(t), os(r, i);
    }
  };
}
function al(e) {
  let t, n, r, i, l = (
    /*i*/
    e[40] !== 0 && jh()
  ), s = (
    /*p*/
    e[38].desc != null && ul(e)
  ), a = (
    /*p*/
    e[38].desc != null && /*progress_level*/
    e[14] && /*progress_level*/
    e[14][
      /*i*/
      e[40]
    ] != null && fl()
  ), o = (
    /*progress_level*/
    e[14] != null && cl(e)
  );
  return {
    c() {
      l && l.c(), t = Oe(), s && s.c(), n = Oe(), a && a.c(), r = Oe(), o && o.c(), i = Pt();
    },
    m(u, f) {
      l && l.m(u, f), N(u, t, f), s && s.m(u, f), N(u, n, f), a && a.m(u, f), N(u, r, f), o && o.m(u, f), N(u, i, f);
    },
    p(u, f) {
      /*p*/
      u[38].desc != null ? s ? s.p(u, f) : (s = ul(u), s.c(), s.m(n.parentNode, n)) : s && (s.d(1), s = null), /*p*/
      u[38].desc != null && /*progress_level*/
      u[14] && /*progress_level*/
      u[14][
        /*i*/
        u[40]
      ] != null ? a || (a = fl(), a.c(), a.m(r.parentNode, r)) : a && (a.d(1), a = null), /*progress_level*/
      u[14] != null ? o ? o.p(u, f) : (o = cl(u), o.c(), o.m(i.parentNode, i)) : o && (o.d(1), o = null);
    },
    d(u) {
      u && (P(t), P(n), P(r), P(i)), l && l.d(u), s && s.d(u), a && a.d(u), o && o.d(u);
    }
  };
}
function jh(e) {
  let t;
  return {
    c() {
      t = V(" /");
    },
    m(n, r) {
      N(n, t, r);
    },
    d(n) {
      n && P(t);
    }
  };
}
function ul(e) {
  let t = (
    /*p*/
    e[38].desc + ""
  ), n;
  return {
    c() {
      n = V(t);
    },
    m(r, i) {
      N(r, n, i);
    },
    p(r, i) {
      i[0] & /*progress*/
      128 && t !== (t = /*p*/
      r[38].desc + "") && Ee(n, t);
    },
    d(r) {
      r && P(n);
    }
  };
}
function fl(e) {
  let t;
  return {
    c() {
      t = V("-");
    },
    m(n, r) {
      N(n, t, r);
    },
    d(n) {
      n && P(t);
    }
  };
}
function cl(e) {
  let t = (100 * /*progress_level*/
  (e[14][
    /*i*/
    e[40]
  ] || 0)).toFixed(1) + "", n, r;
  return {
    c() {
      n = V(t), r = V("%");
    },
    m(i, l) {
      N(i, n, l), N(i, r, l);
    },
    p(i, l) {
      l[0] & /*progress_level*/
      16384 && t !== (t = (100 * /*progress_level*/
      (i[14][
        /*i*/
        i[40]
      ] || 0)).toFixed(1) + "") && Ee(n, t);
    },
    d(i) {
      i && (P(n), P(r));
    }
  };
}
function hl(e) {
  let t, n = (
    /*p*/
    (e[38].desc != null || /*progress_level*/
    e[14] && /*progress_level*/
    e[14][
      /*i*/
      e[40]
    ] != null) && al(e)
  );
  return {
    c() {
      n && n.c(), t = Pt();
    },
    m(r, i) {
      n && n.m(r, i), N(r, t, i);
    },
    p(r, i) {
      /*p*/
      r[38].desc != null || /*progress_level*/
      r[14] && /*progress_level*/
      r[14][
        /*i*/
        r[40]
      ] != null ? n ? n.p(r, i) : (n = al(r), n.c(), n.m(t.parentNode, t)) : n && (n.d(1), n = null);
    },
    d(r) {
      r && P(t), n && n.d(r);
    }
  };
}
function _l(e) {
  let t, n;
  return {
    c() {
      t = De("p"), n = V(
        /*loading_text*/
        e[9]
      ), Le(t, "class", "loading svelte-1txqlrd");
    },
    m(r, i) {
      N(r, t, i), nt(t, n);
    },
    p(r, i) {
      i[0] & /*loading_text*/
      512 && Ee(
        n,
        /*loading_text*/
        r[9]
      );
    },
    d(r) {
      r && P(t);
    }
  };
}
function zh(e) {
  let t, n, r, i, l;
  const s = [Mh, Oh], a = [];
  function o(u, f) {
    return (
      /*status*/
      u[4] === "pending" ? 0 : (
        /*status*/
        u[4] === "error" ? 1 : -1
      )
    );
  }
  return ~(n = o(e)) && (r = a[n] = s[n](e)), {
    c() {
      t = De("div"), r && r.c(), Le(t, "class", i = "wrap " + /*variant*/
      e[8] + " " + /*show_progress*/
      e[6] + " svelte-1txqlrd"), ve(t, "hide", !/*status*/
      e[4] || /*status*/
      e[4] === "complete" || /*show_progress*/
      e[6] === "hidden"), ve(
        t,
        "translucent",
        /*variant*/
        e[8] === "center" && /*status*/
        (e[4] === "pending" || /*status*/
        e[4] === "error") || /*translucent*/
        e[11] || /*show_progress*/
        e[6] === "minimal"
      ), ve(
        t,
        "generating",
        /*status*/
        e[4] === "generating"
      ), ve(
        t,
        "border",
        /*border*/
        e[12]
      ), Je(
        t,
        "position",
        /*absolute*/
        e[10] ? "absolute" : "static"
      ), Je(
        t,
        "padding",
        /*absolute*/
        e[10] ? "0" : "var(--size-8) 0"
      );
    },
    m(u, f) {
      N(u, t, f), ~n && a[n].m(t, null), e[31](t), l = !0;
    },
    p(u, f) {
      let c = n;
      n = o(u), n === c ? ~n && a[n].p(u, f) : (r && (as(), At(a[c], 1, 1, () => {
        a[c] = null;
      }), ss()), ~n ? (r = a[n], r ? r.p(u, f) : (r = a[n] = s[n](u), r.c()), Bt(r, 1), r.m(t, null)) : r = null), (!l || f[0] & /*variant, show_progress*/
      320 && i !== (i = "wrap " + /*variant*/
      u[8] + " " + /*show_progress*/
      u[6] + " svelte-1txqlrd")) && Le(t, "class", i), (!l || f[0] & /*variant, show_progress, status, show_progress*/
      336) && ve(t, "hide", !/*status*/
      u[4] || /*status*/
      u[4] === "complete" || /*show_progress*/
      u[6] === "hidden"), (!l || f[0] & /*variant, show_progress, variant, status, translucent, show_progress*/
      2384) && ve(
        t,
        "translucent",
        /*variant*/
        u[8] === "center" && /*status*/
        (u[4] === "pending" || /*status*/
        u[4] === "error") || /*translucent*/
        u[11] || /*show_progress*/
        u[6] === "minimal"
      ), (!l || f[0] & /*variant, show_progress, status*/
      336) && ve(
        t,
        "generating",
        /*status*/
        u[4] === "generating"
      ), (!l || f[0] & /*variant, show_progress, border*/
      4416) && ve(
        t,
        "border",
        /*border*/
        u[12]
      ), f[0] & /*absolute*/
      1024 && Je(
        t,
        "position",
        /*absolute*/
        u[10] ? "absolute" : "static"
      ), f[0] & /*absolute*/
      1024 && Je(
        t,
        "padding",
        /*absolute*/
        u[10] ? "0" : "var(--size-8) 0"
      );
    },
    i(u) {
      l || (Bt(r), l = !0);
    },
    o(u) {
      At(r), l = !1;
    },
    d(u) {
      u && P(t), ~n && a[n].d(), e[31](null);
    }
  };
}
let ln = [], sr = !1;
async function Vh(e, t = !0) {
  if (!(window.__gradio_mode__ === "website" || window.__gradio_mode__ !== "app" && t !== !0)) {
    if (ln.push(e), !sr)
      sr = !0;
    else
      return;
    await Nh(), requestAnimationFrame(() => {
      let n = [0, 0];
      for (let r = 0; r < ln.length; r++) {
        const l = ln[r].getBoundingClientRect();
        (r === 0 || l.top + window.scrollY <= n[0]) && (n[0] = l.top + window.scrollY, n[1] = r);
      }
      window.scrollTo({ top: n[0] - 20, behavior: "smooth" }), sr = !1, ln = [];
    });
  }
}
function Xh(e, t, n) {
  let r, { $$slots: i = {}, $$scope: l } = t, { i18n: s } = t, { eta: a = null } = t, { queue: o = !1 } = t, { queue_position: u } = t, { queue_size: f } = t, { status: c } = t, { scroll_to_output: h = !1 } = t, { timer: _ = !0 } = t, { show_progress: d = "full" } = t, { message: g = null } = t, { progress: S = null } = t, { variant: b = "default" } = t, { loading_text: T = "Loading..." } = t, { absolute: y = !0 } = t, { translucent: m = !1 } = t, { border: v = !1 } = t, { autoscroll: G } = t, F, q = !1, K = 0, M = 0, ie = null, oe = 0, $ = null, H, I = null, X = !0;
  const x = () => {
    n(25, K = performance.now()), n(26, M = 0), q = !0, C();
  };
  function C() {
    requestAnimationFrame(() => {
      n(26, M = (performance.now() - K) / 1e3), q && C();
    });
  }
  function B() {
    n(26, M = 0), q && (q = !1);
  }
  Ih(() => {
    q && B();
  });
  let w = null;
  function E(p) {
    $i[p ? "unshift" : "push"](() => {
      I = p, n(16, I), n(7, S), n(14, $), n(15, H);
    });
  }
  function A(p) {
    $i[p ? "unshift" : "push"](() => {
      F = p, n(13, F);
    });
  }
  return e.$$set = (p) => {
    "i18n" in p && n(1, s = p.i18n), "eta" in p && n(0, a = p.eta), "queue" in p && n(21, o = p.queue), "queue_position" in p && n(2, u = p.queue_position), "queue_size" in p && n(3, f = p.queue_size), "status" in p && n(4, c = p.status), "scroll_to_output" in p && n(22, h = p.scroll_to_output), "timer" in p && n(5, _ = p.timer), "show_progress" in p && n(6, d = p.show_progress), "message" in p && n(23, g = p.message), "progress" in p && n(7, S = p.progress), "variant" in p && n(8, b = p.variant), "loading_text" in p && n(9, T = p.loading_text), "absolute" in p && n(10, y = p.absolute), "translucent" in p && n(11, m = p.translucent), "border" in p && n(12, v = p.border), "autoscroll" in p && n(24, G = p.autoscroll), "$$scope" in p && n(28, l = p.$$scope);
  }, e.$$.update = () => {
    e.$$.dirty[0] & /*eta, old_eta, queue, timer_start*/
    169869313 && (a === null ? n(0, a = ie) : o && n(0, a = (performance.now() - K) / 1e3 + a), a != null && (n(19, w = a.toFixed(1)), n(27, ie = a))), e.$$.dirty[0] & /*eta, timer_diff*/
    67108865 && n(17, oe = a === null || a <= 0 || !M ? null : Math.min(M / a, 1)), e.$$.dirty[0] & /*progress*/
    128 && S != null && n(18, X = !1), e.$$.dirty[0] & /*progress, progress_level, progress_bar, last_progress_level*/
    114816 && (S != null ? n(14, $ = S.map((p) => {
      if (p.index != null && p.length != null)
        return p.index / p.length;
      if (p.progress != null)
        return p.progress;
    })) : n(14, $ = null), $ ? (n(15, H = $[$.length - 1]), I && (H === 0 ? n(16, I.style.transition = "0", I) : n(16, I.style.transition = "150ms", I))) : n(15, H = void 0)), e.$$.dirty[0] & /*status*/
    16 && (c === "pending" ? x() : B()), e.$$.dirty[0] & /*el, scroll_to_output, status, autoscroll*/
    20979728 && F && h && (c === "pending" || c === "complete") && Vh(F, G), e.$$.dirty[0] & /*status, message*/
    8388624, e.$$.dirty[0] & /*timer_diff*/
    67108864 && n(20, r = M.toFixed(1));
  }, [
    a,
    s,
    u,
    f,
    c,
    _,
    d,
    S,
    b,
    T,
    y,
    m,
    v,
    F,
    $,
    H,
    I,
    oe,
    X,
    w,
    r,
    o,
    h,
    g,
    G,
    K,
    M,
    ie,
    l,
    i,
    E,
    A
  ];
}
class us extends yh {
  constructor(t) {
    super(), Ah(
      this,
      t,
      Xh,
      zh,
      Hh,
      {
        i18n: 1,
        eta: 0,
        queue: 21,
        queue_position: 2,
        queue_size: 3,
        status: 4,
        scroll_to_output: 22,
        timer: 5,
        show_progress: 6,
        message: 23,
        progress: 7,
        variant: 8,
        loading_text: 9,
        absolute: 10,
        translucent: 11,
        border: 12,
        autoscroll: 24
      },
      null,
      [-1, -1]
    );
  }
}
const { setContext: Y_, getContext: Zh } = window.__gradio__svelte__internal, Wh = "WORKER_PROXY_CONTEXT_KEY";
function Jh() {
  return Zh(Wh);
}
function Yh(e) {
  return e.host === window.location.host || e.host === "localhost:7860" || e.host === "127.0.0.1:7860" || // Ref: https://github.com/gradio-app/gradio/blob/v3.32.0/js/app/src/Index.svelte#L194
  e.host === "lite.local";
}
async function dl(e) {
  if (e == null)
    return e;
  const t = new URL(e);
  if (!Yh(t) || t.protocol !== "http:" && t.protocol !== "https:")
    return e;
  const n = Jh();
  if (n == null)
    return e;
  const r = t.pathname;
  return n.httpRequest({
    method: "GET",
    path: r,
    headers: {},
    query_string: ""
  }).then((i) => {
    if (i.status !== 200)
      throw new Error(`Failed to get file ${r} from the Wasm worker.`);
    const l = new Blob([i.body], {
      type: i.headers["Content-Type"]
    });
    return URL.createObjectURL(l);
  });
}
const {
  SvelteComponent: Qh,
  append: Kh,
  assign: Or,
  compute_rest_props: ml,
  detach: Fr,
  element: fs,
  empty: $h,
  exclude_internal_props: e_,
  get_spread_update: t_,
  handle_promise: gl,
  init: n_,
  insert: Gr,
  noop: ct,
  safe_not_equal: r_,
  set_attributes: pl,
  set_data: i_,
  set_style: l_,
  src_url_equal: s_,
  text: o_,
  update_await_block_branch: a_
} = window.__gradio__svelte__internal;
function u_(e) {
  let t, n = (
    /*error*/
    e[3].message + ""
  ), r;
  return {
    c() {
      t = fs("p"), r = o_(n), l_(t, "color", "red");
    },
    m(i, l) {
      Gr(i, t, l), Kh(t, r);
    },
    p(i, l) {
      l & /*src*/
      1 && n !== (n = /*error*/
      i[3].message + "") && i_(r, n);
    },
    d(i) {
      i && Fr(t);
    }
  };
}
function f_(e) {
  let t, n, r = [
    {
      src: n = /*resolved_src*/
      e[2]
    },
    /*$$restProps*/
    e[1]
  ], i = {};
  for (let l = 0; l < r.length; l += 1)
    i = Or(i, r[l]);
  return {
    c() {
      t = fs("img"), pl(t, i);
    },
    m(l, s) {
      Gr(l, t, s);
    },
    p(l, s) {
      pl(t, i = t_(r, [
        s & /*src*/
        1 && !s_(t.src, n = /*resolved_src*/
        l[2]) && { src: n },
        s & /*$$restProps*/
        2 && /*$$restProps*/
        l[1]
      ]));
    },
    d(l) {
      l && Fr(t);
    }
  };
}
function c_(e) {
  return { c: ct, m: ct, p: ct, d: ct };
}
function h_(e) {
  let t, n, r = {
    ctx: e,
    current: null,
    token: null,
    hasCatch: !0,
    pending: c_,
    then: f_,
    catch: u_,
    value: 2,
    error: 3
  };
  return gl(n = dl(
    /*src*/
    e[0]
  ), r), {
    c() {
      t = $h(), r.block.c();
    },
    m(i, l) {
      Gr(i, t, l), r.block.m(i, r.anchor = l), r.mount = () => t.parentNode, r.anchor = t;
    },
    p(i, [l]) {
      e = i, r.ctx = e, l & /*src*/
      1 && n !== (n = dl(
        /*src*/
        e[0]
      )) && gl(n, r) || a_(r, e, l);
    },
    i: ct,
    o: ct,
    d(i) {
      i && Fr(t), r.block.d(i), r.token = null, r = null;
    }
  };
}
function __(e, t, n) {
  const r = ["src"];
  let i = ml(t, r), { src: l = void 0 } = t;
  return e.$$set = (s) => {
    t = Or(Or({}, t), e_(s)), n(1, i = ml(t, r)), "src" in s && n(0, l = s.src);
  }, [l, i];
}
class d_ extends Qh {
  constructor(t) {
    super(), n_(this, t, __, h_, r_, { src: 0 });
  }
}
const {
  SvelteComponent: m_,
  attr: g_,
  create_component: p_,
  destroy_component: b_,
  detach: w_,
  element: v_,
  init: y_,
  insert: E_,
  mount_component: S_,
  safe_not_equal: k_,
  toggle_class: at,
  transition_in: T_,
  transition_out: B_
} = window.__gradio__svelte__internal;
function A_(e) {
  let t, n, r;
  return n = new d_({
    props: {
      src: (
        /*samples_dir*/
        e[1] + /*value*/
        e[0]
      ),
      alt: ""
    }
  }), {
    c() {
      t = v_("div"), p_(n.$$.fragment), g_(t, "class", "container svelte-h11ksk"), at(
        t,
        "table",
        /*type*/
        e[2] === "table"
      ), at(
        t,
        "gallery",
        /*type*/
        e[2] === "gallery"
      ), at(
        t,
        "selected",
        /*selected*/
        e[3]
      );
    },
    m(i, l) {
      E_(i, t, l), S_(n, t, null), r = !0;
    },
    p(i, [l]) {
      const s = {};
      l & /*samples_dir, value*/
      3 && (s.src = /*samples_dir*/
      i[1] + /*value*/
      i[0]), n.$set(s), (!r || l & /*type*/
      4) && at(
        t,
        "table",
        /*type*/
        i[2] === "table"
      ), (!r || l & /*type*/
      4) && at(
        t,
        "gallery",
        /*type*/
        i[2] === "gallery"
      ), (!r || l & /*selected*/
      8) && at(
        t,
        "selected",
        /*selected*/
        i[3]
      );
    },
    i(i) {
      r || (T_(n.$$.fragment, i), r = !0);
    },
    o(i) {
      B_(n.$$.fragment, i), r = !1;
    },
    d(i) {
      i && w_(t), b_(n);
    }
  };
}
function C_(e, t, n) {
  let { value: r } = t, { samples_dir: i } = t, { type: l } = t, { selected: s = !1 } = t;
  return e.$$set = (a) => {
    "value" in a && n(0, r = a.value), "samples_dir" in a && n(1, i = a.samples_dir), "type" in a && n(2, l = a.type), "selected" in a && n(3, s = a.selected);
  }, [r, i, l, s];
}
class Q_ extends m_ {
  constructor(t) {
    super(), y_(this, t, C_, A_, k_, {
      value: 0,
      samples_dir: 1,
      type: 2,
      selected: 3
    });
  }
}
const {
  SvelteComponent: H_,
  add_flush_callback: or,
  assign: cs,
  bind: ar,
  binding_callbacks: ur,
  bubble: P_,
  check_outros: hs,
  create_component: Ue,
  destroy_component: Fe,
  detach: kn,
  empty: _s,
  flush: J,
  get_spread_object: ds,
  get_spread_update: ms,
  group_outros: gs,
  init: N_,
  insert: Tn,
  mount_component: Ge,
  safe_not_equal: I_,
  space: ps,
  transition_in: ce,
  transition_out: he
} = window.__gradio__svelte__internal;
function L_(e) {
  let t, n;
  return t = new vl({
    props: {
      visible: (
        /*visible*/
        e[4]
      ),
      variant: (
        /*_image*/
        e[20] === null ? "dashed" : "solid"
      ),
      border_mode: (
        /*dragging*/
        e[21] ? "focus" : "base"
      ),
      padding: !1,
      elem_id: (
        /*elem_id*/
        e[2]
      ),
      elem_classes: (
        /*elem_classes*/
        e[3]
      ),
      height: (
        /*height*/
        e[9] || void 0
      ),
      width: (
        /*width*/
        e[10]
      ),
      allow_overflow: !1,
      container: (
        /*container*/
        e[12]
      ),
      scale: (
        /*scale*/
        e[13]
      ),
      min_width: (
        /*min_width*/
        e[14]
      ),
      $$slots: { default: [F_] },
      $$scope: { ctx: e }
    }
  }), {
    c() {
      Ue(t.$$.fragment);
    },
    m(r, i) {
      Ge(t, r, i), n = !0;
    },
    p(r, i) {
      const l = {};
      i[0] & /*visible*/
      16 && (l.visible = /*visible*/
      r[4]), i[0] & /*_image*/
      1048576 && (l.variant = /*_image*/
      r[20] === null ? "dashed" : "solid"), i[0] & /*dragging*/
      2097152 && (l.border_mode = /*dragging*/
      r[21] ? "focus" : "base"), i[0] & /*elem_id*/
      4 && (l.elem_id = /*elem_id*/
      r[2]), i[0] & /*elem_classes*/
      8 && (l.elem_classes = /*elem_classes*/
      r[3]), i[0] & /*height*/
      512 && (l.height = /*height*/
      r[9] || void 0), i[0] & /*width*/
      1024 && (l.width = /*width*/
      r[10]), i[0] & /*container*/
      4096 && (l.container = /*container*/
      r[12]), i[0] & /*scale*/
      8192 && (l.scale = /*scale*/
      r[13]), i[0] & /*min_width*/
      16384 && (l.min_width = /*min_width*/
      r[14]), i[0] & /*root, sources, label, show_label, streaming, gradio, active_tool, _image, _points, value, dragging, loading_status*/
      16580963 | i[1] & /*$$scope*/
      4096 && (l.$$scope = { dirty: i, ctx: r }), t.$set(l);
    },
    i(r) {
      n || (ce(t.$$.fragment, r), n = !0);
    },
    o(r) {
      he(t.$$.fragment, r), n = !1;
    },
    d(r) {
      Fe(t, r);
    }
  };
}
function O_(e) {
  let t, n;
  return t = new vl({
    props: {
      visible: (
        /*visible*/
        e[4]
      ),
      variant: "solid",
      border_mode: (
        /*dragging*/
        e[21] ? "focus" : "base"
      ),
      padding: !1,
      elem_id: (
        /*elem_id*/
        e[2]
      ),
      elem_classes: (
        /*elem_classes*/
        e[3]
      ),
      height: (
        /*height*/
        e[9] || void 0
      ),
      width: (
        /*width*/
        e[10]
      ),
      allow_overflow: !1,
      container: (
        /*container*/
        e[12]
      ),
      scale: (
        /*scale*/
        e[13]
      ),
      min_width: (
        /*min_width*/
        e[14]
      ),
      $$slots: { default: [G_] },
      $$scope: { ctx: e }
    }
  }), {
    c() {
      Ue(t.$$.fragment);
    },
    m(r, i) {
      Ge(t, r, i), n = !0;
    },
    p(r, i) {
      const l = {};
      i[0] & /*visible*/
      16 && (l.visible = /*visible*/
      r[4]), i[0] & /*dragging*/
      2097152 && (l.border_mode = /*dragging*/
      r[21] ? "focus" : "base"), i[0] & /*elem_id*/
      4 && (l.elem_id = /*elem_id*/
      r[2]), i[0] & /*elem_classes*/
      8 && (l.elem_classes = /*elem_classes*/
      r[3]), i[0] & /*height*/
      512 && (l.height = /*height*/
      r[9] || void 0), i[0] & /*width*/
      1024 && (l.width = /*width*/
      r[10]), i[0] & /*container*/
      4096 && (l.container = /*container*/
      r[12]), i[0] & /*scale*/
      8192 && (l.scale = /*scale*/
      r[13]), i[0] & /*min_width*/
      16384 && (l.min_width = /*min_width*/
      r[14]), i[0] & /*_image, label, show_label, show_download_button, _selectable, show_share_button, gradio, loading_status*/
      1607906 | i[1] & /*$$scope*/
      4096 && (l.$$scope = { dirty: i, ctx: r }), t.$set(l);
    },
    i(r) {
      n || (ce(t.$$.fragment, r), n = !0);
    },
    o(r) {
      he(t.$$.fragment, r), n = !1;
    },
    d(r) {
      Fe(t, r);
    }
  };
}
function M_(e) {
  let t, n;
  return t = new kl({
    props: {
      unpadded_box: !0,
      size: "large",
      $$slots: { default: [D_] },
      $$scope: { ctx: e }
    }
  }), {
    c() {
      Ue(t.$$.fragment);
    },
    m(r, i) {
      Ge(t, r, i), n = !0;
    },
    p(r, i) {
      const l = {};
      i[1] & /*$$scope*/
      4096 && (l.$$scope = { dirty: i, ctx: r }), t.$set(l);
    },
    i(r) {
      n || (ce(t.$$.fragment, r), n = !0);
    },
    o(r) {
      he(t.$$.fragment, r), n = !1;
    },
    d(r) {
      Fe(t, r);
    }
  };
}
function R_(e) {
  let t, n;
  return t = new nu({
    props: {
      i18n: (
        /*gradio*/
        e[19].i18n
      ),
      type: "image",
      mode: "short"
    }
  }), {
    c() {
      Ue(t.$$.fragment);
    },
    m(r, i) {
      Ge(t, r, i), n = !0;
    },
    p(r, i) {
      const l = {};
      i[0] & /*gradio*/
      524288 && (l.i18n = /*gradio*/
      r[19].i18n), t.$set(l);
    },
    i(r) {
      n || (ce(t.$$.fragment, r), n = !0);
    },
    o(r) {
      he(t.$$.fragment, r), n = !1;
    },
    d(r) {
      Fe(t, r);
    }
  };
}
function D_(e) {
  let t, n;
  return t = new wn({}), {
    c() {
      Ue(t.$$.fragment);
    },
    m(r, i) {
      Ge(t, r, i), n = !0;
    },
    i(r) {
      n || (ce(t.$$.fragment, r), n = !0);
    },
    o(r) {
      he(t.$$.fragment, r), n = !1;
    },
    d(r) {
      Fe(t, r);
    }
  };
}
function U_(e) {
  let t, n, r, i, l;
  const s = [R_, M_], a = [];
  function o(u, f) {
    return f[0] & /*sources*/
    65536 && (t = null), t == null && (t = !!/*sources*/
    u[16].includes("upload")), t ? 0 : 1;
  }
  return n = o(e, [-1, -1]), r = a[n] = s[n](e), {
    c() {
      r.c(), i = _s();
    },
    m(u, f) {
      a[n].m(u, f), Tn(u, i, f), l = !0;
    },
    p(u, f) {
      let c = n;
      n = o(u, f), n === c ? a[n].p(u, f) : (gs(), he(a[c], 1, 1, () => {
        a[c] = null;
      }), hs(), r = a[n], r ? r.p(u, f) : (r = a[n] = s[n](u), r.c()), ce(r, 1), r.m(i.parentNode, i));
    },
    i(u) {
      l || (ce(r), l = !0);
    },
    o(u) {
      he(r), l = !1;
    },
    d(u) {
      u && kn(i), a[n].d(u);
    }
  };
}
function F_(e) {
  let t, n, r, i, l, s, a;
  const o = [
    {
      autoscroll: (
        /*gradio*/
        e[19].autoscroll
      )
    },
    { i18n: (
      /*gradio*/
      e[19].i18n
    ) },
    /*loading_status*/
    e[1]
  ];
  let u = {};
  for (let d = 0; d < o.length; d += 1)
    u = cs(u, o[d]);
  t = new us({ props: u });
  function f(d) {
    e[29](d);
  }
  function c(d) {
    e[30](d);
  }
  function h(d) {
    e[31](d);
  }
  let _ = {
    root: (
      /*root*/
      e[8]
    ),
    sources: (
      /*sources*/
      e[16]
    ),
    label: (
      /*label*/
      e[5]
    ),
    show_label: (
      /*show_label*/
      e[6]
    ),
    streaming: (
      /*streaming*/
      e[18]
    ),
    i18n: (
      /*gradio*/
      e[19].i18n
    ),
    $$slots: { default: [U_] },
    $$scope: { ctx: e }
  };
  return (
    /*active_tool*/
    e[22] !== void 0 && (_.active_tool = /*active_tool*/
    e[22]), /*_image*/
    e[20] !== void 0 && (_.value = /*_image*/
    e[20]), /*_points*/
    e[23] !== void 0 && (_.points = /*_points*/
    e[23]), r = new fh({ props: _ }), ur.push(() => ar(r, "active_tool", f)), ur.push(() => ar(r, "value", c)), ur.push(() => ar(r, "points", h)), r.$on(
      "points_change",
      /*points_change_handler*/
      e[32]
    ), r.$on(
      "edit",
      /*edit_handler*/
      e[33]
    ), r.$on(
      "clear",
      /*clear_handler*/
      e[34]
    ), r.$on(
      "stream",
      /*stream_handler*/
      e[35]
    ), r.$on(
      "drag",
      /*drag_handler*/
      e[36]
    ), r.$on(
      "upload",
      /*upload_handler*/
      e[37]
    ), r.$on(
      "select",
      /*select_handler_1*/
      e[38]
    ), r.$on(
      "share",
      /*share_handler_1*/
      e[39]
    ), r.$on(
      "error",
      /*error_handler_2*/
      e[40]
    ), r.$on(
      "click",
      /*click_handler*/
      e[41]
    ), r.$on(
      "error",
      /*error_handler*/
      e[42]
    ), {
      c() {
        Ue(t.$$.fragment), n = ps(), Ue(r.$$.fragment);
      },
      m(d, g) {
        Ge(t, d, g), Tn(d, n, g), Ge(r, d, g), a = !0;
      },
      p(d, g) {
        const S = g[0] & /*gradio, loading_status*/
        524290 ? ms(o, [
          g[0] & /*gradio*/
          524288 && {
            autoscroll: (
              /*gradio*/
              d[19].autoscroll
            )
          },
          g[0] & /*gradio*/
          524288 && { i18n: (
            /*gradio*/
            d[19].i18n
          ) },
          g[0] & /*loading_status*/
          2 && ds(
            /*loading_status*/
            d[1]
          )
        ]) : {};
        t.$set(S);
        const b = {};
        g[0] & /*root*/
        256 && (b.root = /*root*/
        d[8]), g[0] & /*sources*/
        65536 && (b.sources = /*sources*/
        d[16]), g[0] & /*label*/
        32 && (b.label = /*label*/
        d[5]), g[0] & /*show_label*/
        64 && (b.show_label = /*show_label*/
        d[6]), g[0] & /*streaming*/
        262144 && (b.streaming = /*streaming*/
        d[18]), g[0] & /*gradio*/
        524288 && (b.i18n = /*gradio*/
        d[19].i18n), g[0] & /*gradio, sources*/
        589824 | g[1] & /*$$scope*/
        4096 && (b.$$scope = { dirty: g, ctx: d }), !i && g[0] & /*active_tool*/
        4194304 && (i = !0, b.active_tool = /*active_tool*/
        d[22], or(() => i = !1)), !l && g[0] & /*_image*/
        1048576 && (l = !0, b.value = /*_image*/
        d[20], or(() => l = !1)), !s && g[0] & /*_points*/
        8388608 && (s = !0, b.points = /*_points*/
        d[23], or(() => s = !1)), r.$set(b);
      },
      i(d) {
        a || (ce(t.$$.fragment, d), ce(r.$$.fragment, d), a = !0);
      },
      o(d) {
        he(t.$$.fragment, d), he(r.$$.fragment, d), a = !1;
      },
      d(d) {
        d && kn(n), Fe(t, d), Fe(r, d);
      }
    }
  );
}
function G_(e) {
  let t, n, r, i;
  const l = [
    {
      autoscroll: (
        /*gradio*/
        e[19].autoscroll
      )
    },
    { i18n: (
      /*gradio*/
      e[19].i18n
    ) },
    /*loading_status*/
    e[1]
  ];
  let s = {};
  for (let a = 0; a < l.length; a += 1)
    s = cs(s, l[a]);
  return t = new us({ props: s }), r = new zu({
    props: {
      value: (
        /*_image*/
        e[20]
      ),
      label: (
        /*label*/
        e[5]
      ),
      show_label: (
        /*show_label*/
        e[6]
      ),
      show_download_button: (
        /*show_download_button*/
        e[7]
      ),
      selectable: (
        /*_selectable*/
        e[11]
      ),
      show_share_button: (
        /*show_share_button*/
        e[15]
      ),
      i18n: (
        /*gradio*/
        e[19].i18n
      )
    }
  }), r.$on(
    "select",
    /*select_handler*/
    e[26]
  ), r.$on(
    "share",
    /*share_handler*/
    e[27]
  ), r.$on(
    "error",
    /*error_handler_1*/
    e[28]
  ), {
    c() {
      Ue(t.$$.fragment), n = ps(), Ue(r.$$.fragment);
    },
    m(a, o) {
      Ge(t, a, o), Tn(a, n, o), Ge(r, a, o), i = !0;
    },
    p(a, o) {
      const u = o[0] & /*gradio, loading_status*/
      524290 ? ms(l, [
        o[0] & /*gradio*/
        524288 && {
          autoscroll: (
            /*gradio*/
            a[19].autoscroll
          )
        },
        o[0] & /*gradio*/
        524288 && { i18n: (
          /*gradio*/
          a[19].i18n
        ) },
        o[0] & /*loading_status*/
        2 && ds(
          /*loading_status*/
          a[1]
        )
      ]) : {};
      t.$set(u);
      const f = {};
      o[0] & /*_image*/
      1048576 && (f.value = /*_image*/
      a[20]), o[0] & /*label*/
      32 && (f.label = /*label*/
      a[5]), o[0] & /*show_label*/
      64 && (f.show_label = /*show_label*/
      a[6]), o[0] & /*show_download_button*/
      128 && (f.show_download_button = /*show_download_button*/
      a[7]), o[0] & /*_selectable*/
      2048 && (f.selectable = /*_selectable*/
      a[11]), o[0] & /*show_share_button*/
      32768 && (f.show_share_button = /*show_share_button*/
      a[15]), o[0] & /*gradio*/
      524288 && (f.i18n = /*gradio*/
      a[19].i18n), r.$set(f);
    },
    i(a) {
      i || (ce(t.$$.fragment, a), ce(r.$$.fragment, a), i = !0);
    },
    o(a) {
      he(t.$$.fragment, a), he(r.$$.fragment, a), i = !1;
    },
    d(a) {
      a && kn(n), Fe(t, a), Fe(r, a);
    }
  };
}
function q_(e) {
  let t, n, r, i;
  const l = [O_, L_], s = [];
  function a(o, u) {
    return (
      /*interactive*/
      o[17] ? 1 : 0
    );
  }
  return t = a(e), n = s[t] = l[t](e), {
    c() {
      n.c(), r = _s();
    },
    m(o, u) {
      s[t].m(o, u), Tn(o, r, u), i = !0;
    },
    p(o, u) {
      let f = t;
      t = a(o), t === f ? s[t].p(o, u) : (gs(), he(s[f], 1, 1, () => {
        s[f] = null;
      }), hs(), n = s[t], n ? n.p(o, u) : (n = s[t] = l[t](o), n.c()), ce(n, 1), n.m(r.parentNode, r));
    },
    i(o) {
      i || (ce(n), i = !0);
    },
    o(o) {
      he(n), i = !1;
    },
    d(o) {
      o && kn(r), s[t].d(o);
    }
  };
}
function x_(e) {
  let t, n = e[0], r = 1;
  for (; r < e.length; ) {
    const i = e[r], l = e[r + 1];
    if (r += 2, (i === "optionalAccess" || i === "optionalCall") && n == null)
      return;
    i === "access" || i === "optionalAccess" ? (t = n, n = l(n)) : (i === "call" || i === "optionalCall") && (n = l((...s) => n.call(t, ...s)), t = void 0);
  }
  return n;
}
function j_(e, t, n) {
  let r, i, l, { elem_id: s = "" } = t, { elem_classes: a = [] } = t, { visible: o = !0 } = t, { value: u = null } = t, { label: f } = t, { show_label: c } = t, { show_download_button: h } = t, { root: _ } = t, { proxy_url: d } = t, { height: g } = t, { width: S } = t, { _selectable: b = !1 } = t, { container: T = !0 } = t, { scale: y = null } = t, { min_width: m = void 0 } = t, { loading_status: v } = t, { show_share_button: G = !1 } = t, { sources: F = ["upload"] } = t, { interactive: q } = t, { streaming: K } = t, { gradio: M } = t, ie, oe = null;
  const $ = ({ detail: k }) => M.dispatch("select", k), H = ({ detail: k }) => M.dispatch("share", k), I = ({ detail: k }) => M.dispatch("error", k);
  function X(k) {
    oe = k, n(22, oe);
  }
  function x(k) {
    r = k, n(20, r), n(0, u), n(8, _), n(24, d);
  }
  function C(k) {
    i = k, n(23, i), n(0, u);
  }
  const B = ({ detail: k }) => n(0, u.points = k, u), w = () => M.dispatch("edit"), E = () => {
    n(0, u = null), M.dispatch("clear"), M.dispatch("change");
  }, A = () => M.dispatch("stream"), p = ({ detail: k }) => n(21, ie = k), D = ({ detail: k }) => {
    u == null ? n(0, u = { image: k, points: null }) : n(0, u.image = k, u), M.dispatch("upload");
  }, Se = ({ detail: k }) => M.dispatch("select", k), le = ({ detail: k }) => M.dispatch("share", k), ke = ({ detail: k }) => {
    n(1, v), n(1, v.status = "error", v), M.dispatch("error", k);
  }, _e = () => M.dispatch("error", "bad thing happened");
  function j(k) {
    P_.call(this, e, k);
  }
  return e.$$set = (k) => {
    "elem_id" in k && n(2, s = k.elem_id), "elem_classes" in k && n(3, a = k.elem_classes), "visible" in k && n(4, o = k.visible), "value" in k && n(0, u = k.value), "label" in k && n(5, f = k.label), "show_label" in k && n(6, c = k.show_label), "show_download_button" in k && n(7, h = k.show_download_button), "root" in k && n(8, _ = k.root), "proxy_url" in k && n(24, d = k.proxy_url), "height" in k && n(9, g = k.height), "width" in k && n(10, S = k.width), "_selectable" in k && n(11, b = k._selectable), "container" in k && n(12, T = k.container), "scale" in k && n(13, y = k.scale), "min_width" in k && n(14, m = k.min_width), "loading_status" in k && n(1, v = k.loading_status), "show_share_button" in k && n(15, G = k.show_share_button), "sources" in k && n(16, F = k.sources), "interactive" in k && n(17, q = k.interactive), "streaming" in k && n(18, K = k.streaming), "gradio" in k && n(19, M = k.gradio);
  }, e.$$.update = () => {
    e.$$.dirty[0] & /*value, root, proxy_url*/
    16777473 && n(20, r = u && Re(u.image, _, d)), e.$$.dirty[0] & /*value*/
    1 && n(23, i = u && u.points), e.$$.dirty[0] & /*_image*/
    1048576 && n(25, l = x_([r, "optionalAccess", (k) => k.url])), e.$$.dirty[0] & /*url, gradio*/
    34078720 && l && M.dispatch("change");
  }, [
    u,
    v,
    s,
    a,
    o,
    f,
    c,
    h,
    _,
    g,
    S,
    b,
    T,
    y,
    m,
    G,
    F,
    q,
    K,
    M,
    r,
    ie,
    oe,
    i,
    d,
    l,
    $,
    H,
    I,
    X,
    x,
    C,
    B,
    w,
    E,
    A,
    p,
    D,
    Se,
    le,
    ke,
    _e,
    j
  ];
}
class K_ extends H_ {
  constructor(t) {
    super(), N_(
      this,
      t,
      j_,
      q_,
      I_,
      {
        elem_id: 2,
        elem_classes: 3,
        visible: 4,
        value: 0,
        label: 5,
        show_label: 6,
        show_download_button: 7,
        root: 8,
        proxy_url: 24,
        height: 9,
        width: 10,
        _selectable: 11,
        container: 12,
        scale: 13,
        min_width: 14,
        loading_status: 1,
        show_share_button: 15,
        sources: 16,
        interactive: 17,
        streaming: 18,
        gradio: 19
      },
      null,
      [-1, -1]
    );
  }
  get elem_id() {
    return this.$$.ctx[2];
  }
  set elem_id(t) {
    this.$$set({ elem_id: t }), J();
  }
  get elem_classes() {
    return this.$$.ctx[3];
  }
  set elem_classes(t) {
    this.$$set({ elem_classes: t }), J();
  }
  get visible() {
    return this.$$.ctx[4];
  }
  set visible(t) {
    this.$$set({ visible: t }), J();
  }
  get value() {
    return this.$$.ctx[0];
  }
  set value(t) {
    this.$$set({ value: t }), J();
  }
  get label() {
    return this.$$.ctx[5];
  }
  set label(t) {
    this.$$set({ label: t }), J();
  }
  get show_label() {
    return this.$$.ctx[6];
  }
  set show_label(t) {
    this.$$set({ show_label: t }), J();
  }
  get show_download_button() {
    return this.$$.ctx[7];
  }
  set show_download_button(t) {
    this.$$set({ show_download_button: t }), J();
  }
  get root() {
    return this.$$.ctx[8];
  }
  set root(t) {
    this.$$set({ root: t }), J();
  }
  get proxy_url() {
    return this.$$.ctx[24];
  }
  set proxy_url(t) {
    this.$$set({ proxy_url: t }), J();
  }
  get height() {
    return this.$$.ctx[9];
  }
  set height(t) {
    this.$$set({ height: t }), J();
  }
  get width() {
    return this.$$.ctx[10];
  }
  set width(t) {
    this.$$set({ width: t }), J();
  }
  get _selectable() {
    return this.$$.ctx[11];
  }
  set _selectable(t) {
    this.$$set({ _selectable: t }), J();
  }
  get container() {
    return this.$$.ctx[12];
  }
  set container(t) {
    this.$$set({ container: t }), J();
  }
  get scale() {
    return this.$$.ctx[13];
  }
  set scale(t) {
    this.$$set({ scale: t }), J();
  }
  get min_width() {
    return this.$$.ctx[14];
  }
  set min_width(t) {
    this.$$set({ min_width: t }), J();
  }
  get loading_status() {
    return this.$$.ctx[1];
  }
  set loading_status(t) {
    this.$$set({ loading_status: t }), J();
  }
  get show_share_button() {
    return this.$$.ctx[15];
  }
  set show_share_button(t) {
    this.$$set({ show_share_button: t }), J();
  }
  get sources() {
    return this.$$.ctx[16];
  }
  set sources(t) {
    this.$$set({ sources: t }), J();
  }
  get interactive() {
    return this.$$.ctx[17];
  }
  set interactive(t) {
    this.$$set({ interactive: t }), J();
  }
  get streaming() {
    return this.$$.ctx[18];
  }
  set streaming(t) {
    this.$$set({ streaming: t }), J();
  }
  get gradio() {
    return this.$$.ctx[19];
  }
  set gradio(t) {
    this.$$set({ gradio: t }), J();
  }
}
export {
  Q_ as BaseExample,
  d_ as BaseImage,
  fh as BaseImageUploader,
  zu as BaseStaticImage,
  X0 as BoxDrawer,
  K_ as default
};
