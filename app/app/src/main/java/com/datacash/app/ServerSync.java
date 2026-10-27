package com.datacash.app;

import android.content.Context;

import org.json.JSONObject;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;

/**
 * 녹화 중 서버와 시각 주고받기 (0.5초마다) → phone_server_sync.csv (보낸 시각, 서버 시각, 받은 시각).
 * 워치도 같은 서버와 주고받으므로, 폰과 워치가 직접 연결되지 않아도 서버를 거쳐 두 시계를 맞춘다.
 * 녹화 시작·끝을 서버에 알려 워치가 같은 세션 id 로 기록하게 한다.
 */
final class ServerSync {
    private final Context ctx;
    private Thread thread;
    private volatile boolean running;
    private BufferedWriter w;

    ServerSync(Context c) {
        ctx = c;
    }

    void start(File dir) throws Exception {
        w = new BufferedWriter(new FileWriter(new File(dir, "phone_server_sync.csv")));
        w.write("device_send_ns,server_ns,device_recv_ns\n");
        running = true;
        String sid = dir.getName();
        thread = new Thread(() -> {
            post("/api/live/start", sid);
            while (running) {
                try {
                    long a = Clock.now();
                    JSONObject o = get("/api/time");
                    long b = Clock.now();
                    if (o != null) synchronized (this) { w.write(a + "," + o.getLong("ns") + "," + b + "\n"); }
                    Thread.sleep(500);
                } catch (Exception ignored) {
                }
            }
            post("/api/live/stop", sid);
        }, "server-sync");
        thread.start();
    }

    void stop() {
        running = false;
        try { thread.join(3000); } catch (InterruptedException ignored) {}
        try { synchronized (this) { w.close(); } } catch (Exception ignored) {}
    }

    private JSONObject get(String path) {
        try {
            HttpURLConnection h = (HttpURLConnection) new URL(Store.server(ctx) + path).openConnection();
            h.setRequestProperty("X-DC-Token", Store.token(ctx));
            h.setConnectTimeout(1500);
            h.setReadTimeout(1500);
            if (h.getResponseCode() != 200) return null;
            return new JSONObject(new String(h.getInputStream().readAllBytes(), "UTF-8"));
        } catch (Exception e) {
            return null;
        }
    }

    private void post(String path, String sid) {
        try {
            HttpURLConnection h = (HttpURLConnection) new URL(Store.server(ctx) + path).openConnection();
            h.setRequestMethod("POST");
            h.setRequestProperty("X-DC-Token", Store.token(ctx));
            h.setRequestProperty("Content-Type", "application/json");
            h.setDoOutput(true);
            h.setConnectTimeout(2000);
            try (OutputStream o = h.getOutputStream()) { o.write(new JSONObject().put("sid", sid).toString().getBytes("UTF-8")); }
            h.getResponseCode();
        } catch (Exception ignored) {
        }
    }
}
