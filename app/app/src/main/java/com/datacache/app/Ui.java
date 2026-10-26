package com.datacache.app;

import android.app.Activity;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.view.Gravity;
import android.view.View;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

/** 화면 공통: 데이터캐시 팔레트 (Vibrant Cache · Earn Gold · Charcoal) 와 카드·칩·버튼. */
final class Ui {
    static final int PRIMARY = Color.rgb(0xFF, 0x5F, 0x5F);   // Vibrant Cache
    static final int PRIMARY_DARK = Color.rgb(0xE8, 0x46, 0x46);
    static final int PRIMARY_SOFT = Color.rgb(0xFF, 0xEC, 0xEC);
    static final int ACCENT = Color.rgb(0xFF, 0xAF, 0x5F);    // Earn Gold
    static final int ACCENT_INK = Color.rgb(0xD9, 0x7F, 0x1F);
    static final int ACCENT_SOFT = Color.rgb(0xFF, 0xF3, 0xE3);
    static final int INK = Color.rgb(0x33, 0x33, 0x33);       // Charcoal
    static final int SUB = Color.rgb(0x6E, 0x6E, 0x73);
    static final int LINE = Color.rgb(0xEC, 0xEC, 0xEE);
    static final int BG = Color.rgb(0xFA, 0xFA, 0xFB);
    static final int OK = Color.rgb(0x16, 0xA3, 0x6B);
    static final int OK_SOFT = Color.rgb(0xE6, 0xF7, 0xEF);

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

    static TextView text(Activity a, String s, float sp, int color, boolean bold) {
        TextView t = new TextView(a);
        t.setText(s);
        t.setTextSize(sp);
        t.setTextColor(color);
        if (bold) t.setTypeface(Typeface.DEFAULT_BOLD);
        t.setLineSpacing(0, 1.15f);
        return t;
    }

    /** 스크롤되는 페이지 + 제목 줄 (로고·제목·뒤로). 내용을 넣을 세로 레이아웃을 돌려준다. */
    static LinearLayout page(Activity a, String title, boolean back) {
        ScrollView sv = new ScrollView(a);
        sv.setBackgroundColor(BG);
        sv.setFillViewport(true);
        LinearLayout body = new LinearLayout(a);
        body.setOrientation(LinearLayout.VERTICAL);
        body.setPadding(dp(a, 18), dp(a, 14), dp(a, 18), dp(a, 28));
        sv.addView(body);
        LinearLayout bar = new LinearLayout(a);
        bar.setGravity(Gravity.CENTER_VERTICAL);
        bar.setPadding(0, dp(a, 4), 0, dp(a, 14));
        if (back) {
            TextView b = text(a, "‹", 30, INK, false);
            b.setPadding(0, 0, dp(a, 12), dp(a, 4));
            b.setOnClickListener(v -> a.finish());
            bar.addView(b);
        } else {
            ImageView logo = new ImageView(a);
            logo.setImageResource(R.drawable.logo);
            bar.addView(logo, new LinearLayout.LayoutParams(dp(a, 34), dp(a, 34)));
            View gap = new View(a);
            bar.addView(gap, new LinearLayout.LayoutParams(dp(a, 10), 1));
        }
        bar.addView(text(a, title, 21, INK, true));
        body.addView(bar);
        a.setContentView(sv);
        return body;
    }

    static LinearLayout card(Activity a) {
        LinearLayout c = new LinearLayout(a);
        c.setOrientation(LinearLayout.VERTICAL);
        c.setBackground(round(Color.WHITE, dp(a, 18)));
        c.setPadding(dp(a, 18), dp(a, 16), dp(a, 18), dp(a, 16));
        c.setElevation(dp(a, 1));
        return c;
    }

    static void add(Activity a, LinearLayout parent, View v) {
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(-1, -2);
        lp.bottomMargin = dp(a, 12);
        parent.addView(v, lp);
    }

    static TextView chip(Activity a, String s, int fg, int bg) {
        TextView t = text(a, s, 12, fg, true);
        t.setBackground(round(bg, dp(a, 99)));
        t.setPadding(dp(a, 10), dp(a, 4), dp(a, 10), dp(a, 4));
        return t;
    }

    static TextView button(Activity a, String s, boolean primary, View.OnClickListener l) {
        TextView t = text(a, s, 16, primary ? Color.WHITE : PRIMARY_DARK, true);
        t.setGravity(Gravity.CENTER);
        t.setBackground(primary ? gradient(dp(a, 14)) : round(PRIMARY_SOFT, dp(a, 14)));
        t.setPadding(0, dp(a, 15), 0, dp(a, 15));
        t.setOnClickListener(l);
        return t;
    }

    static LinearLayout.LayoutParams weight() {
        return new LinearLayout.LayoutParams(0, -2, 1f);
    }
}
