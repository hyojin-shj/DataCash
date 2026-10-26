package com.datacache.app;

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

    private static HttpURLConnection open(Context c, String method, String path) throws Exception {
        HttpURLConnection h = (HttpURLConnection) new URL(Store.server(c) + path).openConnection();
        h.setRequestMethod(method);
        h.setRequestProperty("X-DC-Token", Store.token(c));
        h.setConnectTimeout(8000);
        h.setReadTimeout(60000);
        return h;
    }

    private static JSONObject json(HttpURLConnection h) throws Exception {
        int code = h.getResponseCode();
        InputStream in = code < 400 ? h.getInputStream() : h.getErrorStream();
        String s = in == null ? "{}" : new String(in.readAllBytes(), "UTF-8");
        if (code >= 400) throw new Exception("서버 " + code + ": " + s);
        return s.trim().startsWith("[") ? new JSONObject().put("list", new JSONArray(s)) : new JSONObject(s);
    }

    static JSONArray missions(Context c) throws Exception {
        return json(open(c, "GET", "/api/missions")).getJSONArray("missions");
    }

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
                    p.on("올리는 중", (int) (100 * done / Math.max(1, total)));
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

    static String stateKo(String s) {
        switch (s) {
            case "uploading": return "올리는 중";
            case "queued": case "processing": return "처리 중";
            case "done": return "승인";
            case "rejected": return "반려";
            case "failed": return "실패";
            case "upload_failed": return "다시 올리기";
            default: return "저장됨";
        }
    }
}
