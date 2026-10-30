package com.datacash.app;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Context;
import android.content.res.Configuration;
import android.widget.TextView;

import java.util.Locale;

/**
 * 화면 언어: English(기본) · 한국어 · 日本語 · 简体中文 · Español. 고른 값은 Store 의 "lang" 에.
 * 모든 화면(BaseActivity)과 ProcessService 가 attachBaseContext 에서 wrap 으로 감싼다.
 */
final class Lang {
    private Lang() {}

    /** Store "lang" 값 (서버 머리글 X-DC-Lang 에도 그대로). 순서는 R.array.lang_names 와 같다. */
    static final String[] CODES = {"en", "ko", "ja", "zh-CN", "es"};

    static Locale locale(String code) {
        switch (code) {
            case "ko": return Locale.KOREAN;
            case "ja": return Locale.JAPANESE;
            case "zh-CN": return Locale.SIMPLIFIED_CHINESE;
            case "es": return new Locale("es");
            default: return Locale.ENGLISH;
        }
    }

    static Context wrap(Context base) {
        Locale l = locale(Store.lang(base));
        Locale.setDefault(l);
        Configuration cfg = new Configuration(base.getResources().getConfiguration());
        cfg.setLocale(l);
        cfg.setLayoutDirection(l);
        return base.createConfigurationContext(cfg);
    }

    static int index(Context c) {
        String cur = Store.lang(c);
        for (int i = 0; i < CODES.length; i++) if (CODES[i].equals(cur)) return i;
        return 0;
    }

    /** 언어 고르기 창: 다섯 언어를 각자의 이름으로. 바꾸면 지금 화면을 다시 만든다 (다른 화면은 돌아올 때 BaseActivity 가). */
    static void pick(Activity a) {
        String[] names = a.getResources().getStringArray(R.array.lang_names);
        new AlertDialog.Builder(a).setTitle(R.string.lang_title)
                .setSingleChoiceItems(names, index(a), (d, w) -> {
                    d.dismiss();
                    if (!CODES[w].equals(Store.lang(a))) {
                        Store.setLang(a, CODES[w]);
                        a.recreate();
                    }
                })
                .setNegativeButton(R.string.cancel, null).show();
    }

    /** 머리줄 오른쪽 등에 두는 언어 단추 (지금 언어 이름 + ▾). */
    static TextView button(Activity a) {
        String[] names = a.getResources().getStringArray(R.array.lang_names);
        TextView t = Ui.chip(a, names[index(a)] + " ▾", Ui.INK, Ui.FIELD);
        t.setTextSize(13);
        t.setContentDescription(a.getString(R.string.lang_title));
        t.setBackground(Ui.pressable(android.graphics.Color.WHITE, Ui.dp(a, 99)));
        t.setOnClickListener(v -> pick(a));
        return t;
    }
}
