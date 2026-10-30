package com.datacash.app;

import android.content.Context;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;

/** 서버와 주고받기 (머리글 X-DC-Token). 녹화 파일을 하나씩 올리고 처리를 요청, 상태를 받아 upload.json 에 적는다. */
final class Uploader {
    private Uploader() {}

    static final String[] FILES = {"meta.json", "video.mp4", "video_frames.csv", "audio.wav", "audio.json", "phone_imu.csv",
            "beacon.csv", "watch_imu.csv", "watch_audio.wav", "watch_audio.json", "watch_vib.csv", "watch_hr.csv",
            "watch_skin.csv", "clock_sync.csv", "phone_server_sync.csv"};

    interface Progress { void on(String step, int pct); }

    /** 서버가 오류 코드로 답했을 때 (네트워크 실패와 구별). 글은 "HTTP 코드: 본문". */
    static final class HttpError extends Exception {
        final int code;
        HttpError(int code, String body) { super("HTTP " + code + ": " + body); this.code = code; }
    }

    private static HttpURLConnection open(Context c, String method, String path) throws Exception {
        HttpURLConnection h = (HttpURLConnection) new URL(Store.server(c) + path).openConnection();
        h.setRequestMethod(method);
        h.setRequestProperty("X-DC-Token", Store.token(c));
        h.setRequestProperty("X-DC-Lang", Store.lang(c));        // 서버가 미션 글·안내를 이 언어로
        h.setConnectTimeout(8000);
        h.setReadTimeout(60000);
        return h;
    }

    private static JSONObject json(HttpURLConnection h) throws Exception {
        int code = h.getResponseCode();
        InputStream in = code < 400 ? h.getInputStream() : h.getErrorStream();
        String s = in == null ? "{}" : new String(in.readAllBytes(), "UTF-8");
        if (code >= 400) throw new HttpError(code, s);
        return s.trim().startsWith("[") ? new JSONObject().put("list", new JSONArray(s)) : new JSONObject(s);
    }

    /** 기기 id 머리글을 붙여 부른다 (얼굴 확인·동의 인원·이의 신청). body 가 null 이면 본문 없이. */
    static JSONObject call(Context c, String method, String path, byte[] body, String type) throws Exception {
        HttpURLConnection h = open(c, method, path);
        h.setRequestProperty("X-DC-Device", Store.deviceId(c));
        if (body != null) {
            h.setDoOutput(true);
            h.setRequestProperty("Content-Type", type);
            h.setFixedLengthStreamingMode(body.length);
            try (OutputStream o = h.getOutputStream()) { o.write(body); }
        }
        return json(h);
    }

    static JSONObject callJson(Context c, String method, String path, JSONObject body) throws Exception {
        return call(c, method, path, body == null ? null : body.toString().getBytes("UTF-8"), "application/json");
    }

    /** 서버 오류 글에서 사람이 읽을 부분만 ({"detail": "..."}). */
    static String reason(Exception e) {
        String m = String.valueOf(e.getMessage());
        int i = m.indexOf("{");
        try { return i >= 0 ? new JSONObject(m.substring(i)).optString("detail", m) : m; } catch (Exception x) { return m; }
    }

    /** 서버 미션 목록 (내 연령대·성별에 맞는 의뢰 미션 포함) → 폰에 저장. 바뀌었으면 changed=true. */
    static JSONArray missions(Context c) throws Exception {
        HttpURLConnection h = open(c, "GET", "/api/missions");
        h.setRequestProperty("X-DC-Age", java.net.URLEncoder.encode(Store.ageBand(c), "UTF-8"));
        if (!Store.gender(c).isEmpty()) h.setRequestProperty("X-DC-Gender", Store.gender(c));
        JSONObject j = json(h);
        changed = Store.saveCatalog(c, j);
        return j.getJSONArray("missions");
    }

    static volatile boolean changed;

    static void upload(Context c, File dir, Progress p) throws Exception {
        long total = 0, done = 0;
        for (String n : FILES) { File f = new File(dir, n); if (f.exists()) total += f.length(); }
        save(dir, "state", "uploading");
        for (String n : FILES) {
            File f = new File(dir, n);
            if (!f.exists()) continue;
            HttpURLConnection h = open(c, "PUT", "/api/sessions/" + dir.getName() + "/files/" + n);
            h.setDoOutput(true);
            h.setFixedLengthStreamingMode(f.length());
            try (OutputStream o = h.getOutputStream(); FileInputStream in = new FileInputStream(f)) {
                byte[] b = new byte[1 << 16];
                int k;
                while ((k = in.read(b)) > 0) {
                    o.write(b, 0, k);
                    done += k;
                    p.on(c.getString(R.string.state_uploading), (int) (100 * done / Math.max(1, total)));
                }
            }
            json(h);
        }
        merge(dir, json(open(c, "POST", "/api/sessions/" + dir.getName() + "/process")));
    }

    static JSONObject status(Context c, File dir) throws Exception {
        JSONObject st = json(open(c, "GET", "/api/sessions/" + dir.getName() + "/status"));
        merge(dir, st);
        return st;
    }

    /** 아직 끝나지 않은 녹화(처리 중·이의 검토·계좌 기다림)만 서버 상태를 다시 받는다. */
    static void refreshPending(Context c) {
        for (File d : Store.sessions(c)) {
            JSONObject st = localState(d);
            String s = st.optString("state");
            JSONObject py = st.optJSONObject("payout");
            boolean open = s.equals("queued") || s.equals("processing") || s.equals("uploading")
                    || (s.equals("done") && (py == null || py.optString("status").equals("waiting_account")));
            if (!open) continue;
            try { status(c, d); } catch (Exception ignored) { return; }      // 서버가 안 되면 다음에
        }
    }

    /** 내 녹화들의 상태 요약 (바뀌었는지 보는 데만 쓴다). */
    static String signature(Context c) {
        StringBuilder b = new StringBuilder();
        for (File d : Store.sessions(c)) {
            JSONObject st = localState(d);
            JSONObject py = st.optJSONObject("payout");
            b.append(d.getName()).append(st.optString("state")).append(st.optInt("reward")).append(st.optInt("appeal_extra"))
                    .append(py == null ? "" : py.optString("status")).append(';');
        }
        return b.toString();
    }

    static JSONObject localState(File dir) {
        try {
            File f = new File(dir, "upload.json");
            JSONObject o = f.exists() ? new JSONObject(Files.read(f)) : new JSONObject();
            File m = new File(dir, "meta.json");
            if (m.exists()) o.put("title", new JSONObject(Files.read(m)).getJSONObject("mission").optString("title"));
            return o;
        } catch (Exception e) {
            return new JSONObject();
        }
    }

    static void save(File dir, String k, Object v) {
        try { merge(dir, new JSONObject().put(k, v)); } catch (Exception ignored) {}
    }

    static synchronized void merge(File dir, JSONObject add) throws Exception {
        File f = new File(dir, "upload.json");
        JSONObject o = f.exists() ? new JSONObject(Files.read(f)) : new JSONObject();
        for (java.util.Iterator<String> it = add.keys(); it.hasNext(); ) { String k = it.next(); o.put(k, add.get(k)); }
        Files.write(f, o.toString(2));
    }

    static String stateLabel(Context c, String s) {
        switch (s) {
            case "uploading": return c.getString(R.string.state_uploading);
            case "queued": case "processing": return c.getString(R.string.state_processing);
            case "done": return c.getString(R.string.state_done);
            case "rejected": return c.getString(R.string.state_rejected);
            case "failed": return c.getString(R.string.state_failed);
            case "upload_failed": return c.getString(R.string.upload_again);
            case "collected": return c.getString(R.string.state_collected);
            default: return c.getString(R.string.state_saved);
        }
    }
}
