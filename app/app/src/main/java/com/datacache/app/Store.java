package com.datacache.app;

import android.content.Context;
import android.content.SharedPreferences;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;

/** 설정·동의·적립 (폰 안에만). 서버 주소·토큰은 설정 화면에서. */
final class Store {
    private Store() {}

    private static SharedPreferences p(Context c) {
        return c.getSharedPreferences("datacache", Context.MODE_PRIVATE);
    }

    static String server(Context c) { return p(c).getString("server", "http://192.168.0.10:8800"); }
    static String token(Context c) { return p(c).getString("token", ""); }
    static void setServer(Context c, String url, String token) {
        p(c).edit().putString("server", url.trim().replaceAll("/+$", "")).putString("token", token.trim()).apply();
    }

    static boolean consented(Context c) { return p(c).getBoolean("consent_video", false); }
    static boolean bioConsent(Context c) { return p(c).getBoolean("consent_bio", false); }
    static String ageBand(Context c) { return p(c).getString("age_band", ""); }
    static void setConsent(Context c, boolean video, boolean bio, String ageBand) {
        p(c).edit().putBoolean("consent_video", video).putBoolean("consent_bio", bio).putString("age_band", ageBand)
                .putString("consent_at", Clock.wallIso()).putString("consent_version", "datacache-v1").apply();
    }

    static JSONObject consentJson(Context c) throws Exception {
        return new JSONObject().put("video", consented(c)).put("bio", bioConsent(c))
                .put("version", p(c).getString("consent_version", "datacache-v1")).put("at", p(c).getString("consent_at", ""));
    }

    static int earned(Context c) { return p(c).getInt("earned", 0); }
    static void addEarned(Context c, int won) { p(c).edit().putInt("earned", earned(c) + won).apply(); }

    static File sessionsDir(Context c) {
        File d = new File(c.getExternalFilesDir(null), "sessions");
        d.mkdirs();
        return d;
    }

    static String asset(Context c, String name) throws Exception {
        try (InputStream in = c.getAssets().open(name)) {
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            byte[] b = new byte[8192];
            int n;
            while ((n = in.read(b)) > 0) out.write(b, 0, n);
            return out.toString(StandardCharsets.UTF_8.name());
        }
    }

    static JSONArray missions(Context c) throws Exception {
        return new JSONObject(asset(c, "missions.json")).getJSONArray("missions");
    }

    static JSONObject mission(Context c, String id) throws Exception {
        JSONArray ms = missions(c);
        for (int i = 0; i < ms.length(); i++) if (ms.getJSONObject(i).getString("id").equals(id)) return ms.getJSONObject(i);
        return null;
    }
}
