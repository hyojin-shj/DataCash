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
    /** 이 폰의 고정 id (쥐는 힘 보정을 사람·폰마다 저장하는 열쇠). */
    static String deviceId(Context c) {
        String id = p(c).getString("device_id", "");
        if (id.isEmpty()) {
            id = java.util.UUID.randomUUID().toString().substring(0, 12);
            p(c).edit().putString("device_id", id).apply();
        }
        return id;
    }
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

    static JSONObject primitivesKo(Context c) throws Exception {
        return new JSONObject(asset(c, "missions.json")).getJSONObject("primitives_ko");
    }

    /** 오늘의 보너스 미션: 한국 날짜로 하루에 하나, 목록을 차례로 (서버 pricing.today_mission 과 같은 식). */
    static String todayMission(Context c) throws Exception {
        JSONArray ms = missions(c);
        long ordinal = java.time.LocalDate.now(java.time.ZoneId.of("Asia/Seoul")).toEpochDay() + 719163L;
        return ms.getJSONObject((int) (ordinal % ms.length())).getString("id");
    }

    /** 미션 하나 (1분 안팎) 보상 예상: 동의 등급의 분당 보상 (워치 생체 동의 → 생체 등급). */
    static int perMission(Context c) { return bioConsent(c) ? 830 : 550; }

    static java.util.Set<String> doneSituations(Context c, String missionId) {
        return new java.util.HashSet<>(p(c).getStringSet("done_" + missionId, new java.util.HashSet<>()));
    }

    static void markSituation(Context c, String missionId, String situation) {
        java.util.Set<String> s = doneSituations(c, missionId);
        s.add(situation);
        p(c).edit().putStringSet("done_" + missionId, s).apply();
    }

    /** 보상 받을 계좌: 은행과 번호는 이 폰에만 (서버에는 보내지 않는다). */
    static String bank(Context c) { return p(c).getString("payout_bank", ""); }
    static String account(Context c) { return p(c).getString("payout_account", ""); }
    static void setPayout(Context c, String bank, String account) {
        p(c).edit().putString("payout_bank", bank).putString("payout_account", account).apply();
    }
    static String maskedAccount(Context c) {
        String a = account(c);
        return a.length() <= 4 ? a : "••••" + a.substring(a.length() - 4);
    }

    /** 내 녹화 폴더들 (최근 것 먼저). */
    static File[] sessions(Context c) {
        File[] ss = sessionsDir(c).listFiles(File::isDirectory);
        if (ss == null) return new File[0];
        java.util.Arrays.sort(ss, (x, y) -> y.getName().compareTo(x.getName()));
        return ss;
    }

    static JSONObject mission(Context c, String id) throws Exception {
        JSONArray ms = missions(c);
        for (int i = 0; i < ms.length(); i++) if (ms.getJSONObject(i).getString("id").equals(id)) return ms.getJSONObject(i);
        return null;
    }
}
