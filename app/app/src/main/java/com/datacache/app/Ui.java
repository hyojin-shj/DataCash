package com.datacache.app;

import android.app.Activity;
import android.content.res.ColorStateList;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.Drawable;
import android.graphics.drawable.GradientDrawable;
import android.graphics.drawable.RippleDrawable;
import android.text.InputType;
import android.view.Gravity;
import android.view.View;
import android.view.Window;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

/**
 * 화면 공통 틀: 연회색 바탕 위 흰 둥근 카드, 굵은 큰 숫자, 아이콘 칸이 있는 목록 줄, 둥근 채움 버튼, 회색 입력칸, 둥근 뱃지.
 * 색은 데이터캐시 컬러 차트 — Vibrant Cache #FF5F5F · Earn Gold #FFAF5F · White · Charcoal #333333.
 */
final class Ui {
    static final int PRIMARY = Color.rgb(0xFF, 0x5F, 0x5F);   // Vibrant Cache
    static final int PRIMARY_DARK = Color.rgb(0xE8, 0x46, 0x46);
    static final int PRIMARY_SOFT = Color.rgb(0xFF, 0xEC, 0xEC);
    static final int ACCENT = Color.rgb(0xFF, 0xAF, 0x5F);    // Earn Gold
    static final int ACCENT_INK = Color.rgb(0xE8, 0x8A, 0x2E); // 글자로 읽히게 조금 진하게
    static final int ACCENT_SOFT = Color.rgb(0xFF, 0xF2, 0xE2);
    static final int INK = Color.rgb(0x33, 0x33, 0x33);       // Charcoal
    static final int SUB = Color.rgb(0x8B, 0x95, 0xA1);
    static final int LINE = Color.rgb(0xE5, 0xE8, 0xEB);
    static final int LINE_DARK = Color.rgb(0xB0, 0xB8, 0xC1);
    static final int BG = Color.rgb(0xF2, 0xF4, 0xF6);
    static final int FIELD = Color.rgb(0xF2, 0xF4, 0xF6);
    static final int OK = Color.rgb(0x16, 0xA3, 0x6B);
    static final int OK_SOFT = Color.rgb(0xE6, 0xF7, 0xEF);
    static final int VIOLET = Color.rgb(0x7B, 0x61, 0xFF);
    static final int VIOLET_SOFT = Color.rgb(0xF1, 0xEE, 0xFF);

    private Ui() {}

    static int dp(Activity a, float v) {
        return Math.round(v * a.getResources().getDisplayMetrics().density);
    }

    static GradientDrawable round(int color, int radius) {
        GradientDrawable g = new GradientDrawable();
        g.setColor(color);
        g.setCornerRadius(radius);
        return g;
    }

    static GradientDrawable gradient(int radius) {
        GradientDrawable g = new GradientDrawable(GradientDrawable.Orientation.TL_BR, new int[]{PRIMARY, ACCENT});
        g.setCornerRadius(radius);
        return g;
    }

    /** 누르면 살짝 어두워지는 둥근 바탕. */
    static Drawable pressable(int color, int radius) {
        return new RippleDrawable(ColorStateList.valueOf(0x1A000000), round(color, radius), round(Color.BLACK, radius));
    }

    static TextView text(Activity a, String s, float sp, int color, boolean bold) {
        TextView t = new TextView(a);
        t.setText(s);
        t.setTextSize(sp);
        t.setTextColor(color);
        t.setTypeface(Typeface.create("sans-serif", bold ? Typeface.BOLD : Typeface.NORMAL));
        t.setLineSpacing(0, 1.15f);
        t.setIncludeFontPadding(false);
        return t;
    }

    @SuppressWarnings("deprecation")
    static void lightBars(Activity a) {
        Window w = a.getWindow();
        w.setStatusBarColor(BG);
        w.setNavigationBarColor(BG);
        w.getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR | View.SYSTEM_UI_FLAG_LIGHT_NAVIGATION_BAR);
    }

    static LinearLayout page(Activity a, String title, boolean back) {
        return page(a, title, back, null);
    }

    /** 머리줄(뒤로·제목·오른쪽 것) + 회색 바탕 스크롤 본문. 본문 세로 레이아웃을 돌려준다. 제목이 null 이면 로고·이름. */
    static LinearLayout page(Activity a, String title, boolean back, View right) {
        lightBars(a);
        LinearLayout page = new LinearLayout(a);
        page.setOrientation(LinearLayout.VERTICAL);
        page.setBackgroundColor(BG);
        page.addView(topBar(a, title, back, right));
        ScrollView sv = new ScrollView(a);
        sv.setVerticalScrollBarEnabled(false);
        LinearLayout body = new LinearLayout(a);
        body.setOrientation(LinearLayout.VERTICAL);
        body.setPadding(dp(a, 16), dp(a, 6), dp(a, 16), dp(a, 32));
        sv.addView(body);
        page.addView(sv, new LinearLayout.LayoutParams(-1, 0, 1f));
        a.setContentView(page);
        return body;
    }

    private static LinearLayout topBar(Activity a, String title, boolean back, View right) {
        LinearLayout bar = new LinearLayout(a);
        bar.setGravity(Gravity.CENTER_VERTICAL);
        bar.setMinimumHeight(dp(a, 56));
        bar.setPadding(dp(a, back ? 4 : 18), dp(a, 6), dp(a, 16), dp(a, 6));
        if (back) {
            Icon b = new Icon(a, Icon.BACK, INK);
            int p = dp(a, 9);
            b.setPadding(p, p, p, p);
            b.setBackground(pressable(Color.TRANSPARENT, dp(a, 22)));
            b.setContentDescription("뒤로");
            b.setOnClickListener(v -> a.finish());
            bar.addView(b, new LinearLayout.LayoutParams(dp(a, 44), dp(a, 44)));
        }
        if (title == null) {
            LinearLayout brand = new LinearLayout(a);
            brand.setGravity(Gravity.CENTER_VERTICAL);
            android.widget.ImageView logo = new android.widget.ImageView(a);
            logo.setImageResource(R.drawable.logo);
            brand.addView(logo, new LinearLayout.LayoutParams(dp(a, 30), dp(a, 30)));
            TextView n = text(a, "데이터캐시", 20, INK, true);
            n.setPadding(dp(a, 10), 0, 0, 0);
            brand.addView(n);
            bar.addView(brand, weight());
        } else {
            bar.addView(text(a, title, back ? 19 : 22, INK, true), weight());
        }
        if (right != null) bar.addView(right);
        return bar;
    }

    /** 흰 둥근 카드. */
    static LinearLayout card(Activity a) {
        LinearLayout c = new LinearLayout(a);
        c.setOrientation(LinearLayout.VERTICAL);
        c.setBackground(round(Color.WHITE, dp(a, 22)));
        c.setPadding(dp(a, 20), dp(a, 20), dp(a, 20), dp(a, 20));
        return c;
    }

    /** 목록 줄을 담는 카드 (안쪽 여백이 작다). */
    static LinearLayout listCard(Activity a) {
        LinearLayout c = card(a);
        c.setPadding(dp(a, 8), dp(a, 8), dp(a, 8), dp(a, 8));
        return c;
    }

    static void add(Activity a, LinearLayout parent, View v) {
        parent.addView(v, margins(a, -1, -2, 0, 0, 0, 0, 12));
    }

    /** 연한 색 둥근 칸 안의 아이콘. */
    static FrameLayout iconTile(Activity a, int kind, int fg, int bg, int sizeDp) {
        FrameLayout f = new FrameLayout(a);
        f.setBackground(round(bg, dp(a, sizeDp * 0.32f)));
        int in = dp(a, sizeDp * 0.52f);
        f.addView(new Icon(a, kind, fg), new FrameLayout.LayoutParams(in, in, Gravity.CENTER));
        return f;
    }

    /** 목록 한 줄: 아이콘 칸 · 제목/설명 · 오른쪽(뱃지, 없으면 누를 수 있을 때 꺾쇠). */
    static LinearLayout row(Activity a, int kind, int fg, int bg, String title, String desc, View right, View.OnClickListener click) {
        LinearLayout r = new LinearLayout(a);
        r.setGravity(Gravity.CENTER_VERTICAL);
        r.setPadding(dp(a, 12), dp(a, 12), dp(a, 8), dp(a, 12));
        if (kind >= 0) r.addView(iconTile(a, kind, fg, bg, 44), margins(a, dp(a, 44), dp(a, 44), 0, 0, 0, 14, 0));
        LinearLayout words = new LinearLayout(a);
        words.setOrientation(LinearLayout.VERTICAL);
        words.addView(text(a, title, 16, INK, true));
        if (desc != null && !desc.isEmpty()) {
            TextView d = text(a, desc, 13, SUB, false);
            d.setPadding(0, dp(a, 5), 0, 0);
            words.addView(d);
        }
        r.addView(words, weight());
        if (right != null) r.addView(right);
        else if (click != null) r.addView(new Icon(a, Icon.CHEVRON, LINE_DARK), new LinearLayout.LayoutParams(dp(a, 20), dp(a, 20)));
        if (click != null) {
            r.setBackground(pressable(Color.TRANSPARENT, dp(a, 16)));
            r.setOnClickListener(click);
        }
        return r;
    }

    static TextView sectionTitle(Activity a, String s) {
        TextView t = text(a, s, 15, INK, true);
        t.setPadding(dp(a, 4), dp(a, 12), 0, dp(a, 10));
        return t;
    }

    static TextView chip(Activity a, String s, int fg, int bg) {
        TextView t = text(a, s, 12, fg, true);
        t.setBackground(round(bg, dp(a, 99)));
        t.setPadding(dp(a, 10), dp(a, 5), dp(a, 10), dp(a, 5));
        return t;
    }

    /** 둥근 채움 버튼: primary 면 코랄 바탕 흰 글자, 아니면 회색 바탕 진한 글자. */
    static TextView button(Activity a, String s, boolean primary, View.OnClickListener l) {
        TextView t = text(a, s, 16, primary ? Color.WHITE : INK, true);
        t.setGravity(Gravity.CENTER);
        t.setMinHeight(dp(a, 54));
        t.setPadding(dp(a, 16), 0, dp(a, 16), 0);
        t.setBackground(pressable(primary ? PRIMARY : FIELD, dp(a, 16)));
        t.setOnClickListener(l);
        return t;
    }

    static EditText input(Activity a, String hint, String value, int type) {
        EditText e = new EditText(a);
        e.setHint(hint);
        e.setText(value);
        e.setInputType(type == 0 ? InputType.TYPE_CLASS_TEXT : type);
        e.setTextSize(16);
        e.setTextColor(INK);
        e.setHintTextColor(SUB);
        e.setSingleLine(true);
        e.setBackground(round(FIELD, dp(a, 14)));
        e.setPadding(dp(a, 16), dp(a, 15), dp(a, 16), dp(a, 15));
        return e;
    }

    /** 얇은 진행 막대 (0~1). */
    static View progress(Activity a, float frac) {
        FrameLayout f = new FrameLayout(a);
        f.setBackground(round(LINE, dp(a, 3)));
        View fill = new View(a);
        fill.setBackground(round(PRIMARY, dp(a, 3)));
        f.addView(fill, new FrameLayout.LayoutParams(0, -1));
        f.addOnLayoutChangeListener((v, l, t, r, b, ol, ot, or, ob) -> {
            int w = Math.round((r - l) * Math.max(0f, Math.min(1f, frac)));
            if (fill.getLayoutParams().width != w) {
                fill.getLayoutParams().width = w;
                fill.post(fill::requestLayout);
            }
        });
        return f;
    }

    /**
     * 카메라 미리보기를 늘이지 않고 칸 안에 비율 그대로 (찍히는 범위가 다 보이게). TextureView 는 버퍼를 칸에 억지로 늘이고
     * 화면이 가로일 때 버퍼를 돌려 주지 않으므로, 화면 방향에 맞춰 돌리고 줄인다. 버퍼는 1280×720.
     */
    static void fitPreview(Activity a, android.view.TextureView v) {
        if (v.getTag() == null) {
            v.setTag(Boolean.TRUE);
            v.addOnLayoutChangeListener((x, l, t, r, b, ol, ot, or, ob) -> applyFit(a, v));
        }
        applyFit(a, v);
    }

    @SuppressWarnings("deprecation")
    static void applyFit(Activity a, android.view.TextureView v) {
        float vw = v.getWidth(), vh = v.getHeight(), bw = 1280f, bh = 720f;
        if (vw <= 0 || vh <= 0) return;
        int rot = a.getWindowManager().getDefaultDisplay().getRotation();
        float cx = vw / 2, cy = vh / 2;
        android.graphics.Matrix m = new android.graphics.Matrix();
        if (rot == android.view.Surface.ROTATION_90 || rot == android.view.Surface.ROTATION_270) {
            android.graphics.RectF view = new android.graphics.RectF(0, 0, vw, vh);
            android.graphics.RectF buf = new android.graphics.RectF(0, 0, bh, bw);
            buf.offset(cx - buf.centerX(), cy - buf.centerY());
            m.setRectToRect(view, buf, android.graphics.Matrix.ScaleToFit.FILL);
            float k = Math.min(vh / bh, vw / bw);
            m.postScale(k, k, cx, cy);
            m.postRotate(90 * (rot - 2), cx, cy);
        } else {   // 세로: 버퍼가 이미 세워져 칸을 채우므로 비율만
            float k = Math.min(vw / bh, vh / bw);
            m.setScale(k * bh / vw, k * bw / vh, cx, cy);
            if (rot == android.view.Surface.ROTATION_180) m.postRotate(180, cx, cy);
        }
        v.setTransform(m);
    }

    static LinearLayout.LayoutParams weight() {
        return new LinearLayout.LayoutParams(0, -2, 1f);
    }

    static LinearLayout.LayoutParams wrap() {
        return new LinearLayout.LayoutParams(-2, -2);
    }

    static LinearLayout.LayoutParams margins(Activity a, int w, int h, float weight, int l, int t, int r, int b) {
        LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(w, h, weight);
        p.setMargins(dp(a, l), dp(a, t), dp(a, r), dp(a, b));
        return p;
    }
}
