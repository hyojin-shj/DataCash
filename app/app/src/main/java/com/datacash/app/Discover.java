package com.datacash.app;

import android.content.Context;
import android.net.nsd.NsdManager;
import android.net.nsd.NsdServiceInfo;

import java.net.HttpURLConnection;
import java.net.InetAddress;
import java.net.URL;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

/**
 * 서버 찾기 — 서버(맥)가 같은 와이파이에 '_datacash._tcp' 이름을 알린다 (mDNS).
 * 저장된 주소로 서버가 안 닿으면 이 이름으로 다시 찾아 주소를 바꿔 둔다 (토큰은 그대로). 맥 IP 가 바뀌어도 앱을 고칠 필요가 없다.
 * 화면 스레드에서 부르지 않는다 (최대 몇 초 기다린다).
 */
final class Discover {
    private Discover() {}

    static final String SERVICE = "_datacash._tcp.";
    private static final long FIND_MS = 5000;
    private static long lastTry;

    /** 서버가 닿으면 그대로, 안 닿으면 찾아서 바꾼다. 서버에 닿게 됐으면 true. */
    static synchronized boolean ensure(Context c) {
        if (System.currentTimeMillis() - lastTry < 10000) return false;     // 안 될 때 계속 확인·찾기 하지 않게
        if (reachable(Store.server(c))) return true;
        String found = find(c);
        if ((found == null || !reachable(found)) && reachable(BuildConfig.DEFAULT_SERVER)) found = BuildConfig.DEFAULT_SERVER;  // 집 밖: 터널 주소
        if (found == null || !reachable(found)) found = pointer();   // 터널 주소가 바뀌었으면: 저장소의 server.txt 가 지금 주소
        if (found == null || !reachable(found)) {
            lastTry = System.currentTimeMillis();
            return false;
        }
        Store.setServer(c, found, Store.token(c));
        return true;
    }

    /** 공개 저장소의 server.txt 첫 줄 (터널을 다시 열면 이 파일만 고친다). 못 읽으면 null. */
    static String pointer() {
        if (BuildConfig.SERVER_POINTER.isEmpty()) return null;
        try {
            HttpURLConnection h = (HttpURLConnection) new URL(BuildConfig.SERVER_POINTER).openConnection();
            h.setConnectTimeout(3000);
            h.setReadTimeout(3000);
            if (h.getResponseCode() != 200) return null;
            java.io.BufferedReader r = new java.io.BufferedReader(new java.io.InputStreamReader(h.getInputStream()));
            String s = r.readLine();
            r.close();
            h.disconnect();
            s = s == null ? "" : s.trim();
            return s.startsWith("https://") ? s.replaceAll("/+$", "") : null;
        } catch (Exception e) {
            return null;
        }
    }

    static boolean reachable(String base) {
        try {
            HttpURLConnection h = (HttpURLConnection) new URL(base + "/api/time").openConnection();
            h.setConnectTimeout(4000);
            h.setReadTimeout(4000);
            boolean ok = h.getResponseCode() < 500;
            h.disconnect();
            return ok;
        } catch (Exception e) {
            return false;
        }
    }

    /** mDNS 로 DataCash 서버를 찾아 "http://ip:port" (못 찾으면 null). */
    static String find(Context c) {
        NsdManager nsd = (NsdManager) c.getApplicationContext().getSystemService(Context.NSD_SERVICE);
        if (nsd == null) return null;
        CountDownLatch done = new CountDownLatch(1);
        String[] out = {null};
        boolean[] resolving = {false};
        NsdManager.DiscoveryListener listener = new NsdManager.DiscoveryListener() {
            @Override public void onServiceFound(NsdServiceInfo s) {
                synchronized (resolving) {
                    if (resolving[0] || !s.getServiceName().startsWith("DataCash")) return;
                    resolving[0] = true;
                }
                nsd.resolveService(s, new NsdManager.ResolveListener() {
                    @Override public void onServiceResolved(NsdServiceInfo r) {
                        InetAddress a = r.getHost();
                        if (a != null) out[0] = "http://" + a.getHostAddress() + ":" + r.getPort();
                        done.countDown();
                    }
                    @Override public void onResolveFailed(NsdServiceInfo r, int err) {
                        synchronized (resolving) { resolving[0] = false; }
                    }
                });
            }
            @Override public void onServiceLost(NsdServiceInfo s) {}
            @Override public void onDiscoveryStarted(String t) {}
            @Override public void onDiscoveryStopped(String t) {}
            @Override public void onStartDiscoveryFailed(String t, int err) { done.countDown(); }
            @Override public void onStopDiscoveryFailed(String t, int err) {}
        };
        try {
            nsd.discoverServices(SERVICE, NsdManager.PROTOCOL_DNS_SD, listener);
            done.await(FIND_MS, TimeUnit.MILLISECONDS);
        } catch (Exception ignored) {
        } finally {
            try { nsd.stopServiceDiscovery(listener); } catch (Exception ignored) {}
        }
        return out[0];
    }
}
