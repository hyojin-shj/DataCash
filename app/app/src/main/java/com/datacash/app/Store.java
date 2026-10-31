package com.datacash.app;

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
        return c.getSharedPreferences("datacash", Context.MODE_PRIVATE);
    }

    /** 화면 언어 (en · ko · ja · zh-CN · es, 기본 en). */
    static String lang(Context c) { return p(c).getString("lang", "en"); }
    static void setLang(Context c, String code) { p(c).edit().putString("lang", code).commit(); }

    static String server(Context c) { return p(c).getString("server", BuildConfig.DEFAULT_SERVER); }
    static String token(Context c) { String t = p(c).getString("token", ""); return t.isEmpty() ? BuildConfig.DEFAULT_TOKEN : t; }
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
                .putString("consent_at", Clock.wallIso()).putString("consent_version", "datacash-v1").apply();
    }

    static JSONObject consentJson(Context c) throws Exception {
        return new JSONObject().put("video", consented(c)).put("bio", bioConsent(c))
                .put("version", p(c).getString("consent_version", "datacash-v1")).put("at", p(c).getString("consent_at", ""));
    }

    /** 모은 보상: 승인된 녹화의 보상 + 이의 신청으로 인정된 추가 보상 (서버 상태에서). */
    static int earned(Context c) {
        int won = 0;
        for (File d : sessions(c)) {
            JSONObject st = Uploader.localState(d);
            if (st.optString("state").equals("done")) won += st.optInt("reward") + st.optInt("appeal_extra");
        }
        return won;
    }

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

    /** 미션 목록: 서버에서 받은 것(구매 기업 의뢰 미션 포함, 내 조건에 맞는 것만)이 있으면 그것, 없으면 앱에 든 기본 미션. */
    static JSONArray missions(Context c) throws Exception {
        return catalog(c).getJSONArray("missions");
    }

    static JSONObject primitivesKo(Context c) throws Exception {
        JSONObject k = catalog(c).optJSONObject("primitives_ko");
        return k != null && k.length() > 0 ? k : new JSONObject(asset(c, "missions.json")).getJSONObject("primitives_ko");
    }

    /** 녹화 제목을 지금 앱 언어로: 녹화할 때 저장한 제목은 그때 언어라, 미션 id 로 카탈로그(서버 · 없으면 앱에 든 것)에서 다시 찾는다. */
    static String titleOf(Context c, JSONObject st, String fallback) {
        String id = st.optString("mission_id", st.optString("mission", ""));
        if (id.equals("grip_calibration")) return c.getString(R.string.menu_calib);
        String lang = lang(c);
        try {
            for (JSONObject cat : new JSONObject[]{catalog(c), new JSONObject(asset(c, "missions.json"))}) {
                JSONArray ms = cat.getJSONArray("missions");
                for (int i = 0; i < ms.length(); i++) {
                    JSONObject m = ms.getJSONObject(i);
                    if (!m.optString("id").equals(id)) continue;
                    JSONObject tr = m.optJSONObject("i18n") == null ? null : m.optJSONObject("i18n").optJSONObject(lang.equals("zh-CN") ? "zh-CN" : lang);
                    if (tr != null && tr.has("title")) return tr.optString("title");
                    String t = m.optString("title");
                    if (lang.equals("ko") || !t.matches(".*[\\uac00-\\ud7a3].*")) return t;
                    if (m.has("title_en")) return m.optString("title_en");
                }
            }
        } catch (Exception ignored) {}
        return st.optString("title", fallback);
    }

    /** 상황 i 의 보이는 이름: 서버가 준 situation_labels, 없으면 앱에 든 i18n.en (한국어 화면은 원문). 상황 원문은 완료 표시·녹화 메타의 열쇠라 그대로 둔다. */
    static String situationLabel(Context c, JSONObject m, int i) {
        String s = m.optJSONArray("situations").optString(i);
        if (lang(c).equals("ko")) return s;
        JSONArray l = m.optJSONArray("situation_labels");
        if (l == null) {
            try {
                JSONArray ms = new JSONObject(asset(c, "missions.json")).getJSONArray("missions");
                for (int k = 0; k < ms.length(); k++)
                    if (ms.getJSONObject(k).optString("id").equals(m.optString("id")))
                        l = ms.getJSONObject(k).getJSONObject("i18n").getJSONObject("en").optJSONArray("situations");
            } catch (Exception ignored) {}
        }
        return l != null && l.length() > i ? l.optString(i, s) : s;
    }

    private static File cacheFile(Context c) { return new File(c.getFilesDir(), "missions_cache.json"); }

    static JSONObject catalog(Context c) throws Exception {
        File f = cacheFile(c);
        if (f.exists()) {
            try { return new JSONObject(Files.read(f)); } catch (Exception ignored) {}
        }
        return new JSONObject(asset(c, "missions.json"));
    }

    /** 서버 미션 목록을 저장. 바뀌었으면 true. */
    static boolean saveCatalog(Context c, JSONObject j) {
        try {
            String s = j.toString();
            File f = cacheFile(c);
            if (f.exists() && Files.read(f).equals(s)) return false;
            Files.write(f, s);
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    static String gender(Context c) { return p(c).getString("gender", ""); }
    static void setGender(Context c, String g) { p(c).edit().putString("gender", g).apply(); }

    /** 오늘의 보너스 미션: 한국 날짜로 하루에 하나, 기본 미션을 차례로 (서버 pricing.today_mission 과 같은 식). */
    static String todayMission(Context c) throws Exception {
        JSONArray ms = new JSONObject(asset(c, "missions.json")).getJSONArray("missions");
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

    static boolean ownerFace(Context c) { return p(c).getBoolean("owner_face", false); }
    static void setOwnerFace(Context c, boolean on) { p(c).edit().putBoolean("owner_face", on).apply(); }

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
