package com.datacash.wear;

import android.content.Context;
import android.os.SystemClock;

import org.json.JSONArray;
import org.json.JSONObject;

import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.util.ArrayList;
import java.util.List;

/**
 * Samsung Health Sensor SDK 다리 — 갤럭시 워치의 연속 심박(+ 박동 간격)과 피부온도.
 * SDK(.aar)는 삼성 개발자 사이트에서 받아 app/wear/libs/ 에 넣는다 (계정·약관 동의가 필요해 저장소에 넣지 않는다).
 * SDK 가 없어도 빌드되도록 리플렉션으로 부른다: 없으면 available() == false → 안드로이드 기본 심박 센서만 쓴다.
 * 파트너 등록 전에는 워치의 Health Platform 을 개발자 모드로 켜야 연결된다 (docs/samsung_health_sdk.md).
 *
 * 받은 값의 시각: SDK 는 벽시계(ms)로 준다 → 받은 순간의 (BOOTTIME − 벽시계) 차이로 BOOTTIME(ns) 으로 옮긴다.
 */
final class SamsungHealth {
    static final String PKG = "com.samsung.android.service.health.tracking";
    static final int NO_STATUS = -1;     // 이 SDK 버전에 측정 상태 값이 없음 (피부온도: 1.4.1)

    interface Sink {
        void heartRate(long tNs, int bpm, int status);
        void ibi(long tNs, int ms, int status);
        void skin(long tNs, float objectC, float ambientC, int status);
        void state(String what);
    }

    private final Context ctx;
    private final Sink sink;
    private Object service;
    private final List<Object> trackers = new ArrayList<>();
    final JSONObject info = new JSONObject();

    SamsungHealth(Context c, Sink s) {
        ctx = c;
        sink = s;
    }

    static boolean available() {
        try {
            Class.forName(PKG + ".HealthTrackingService");
            return true;
        } catch (Throwable t) {
            return false;
        }
    }

    /** 연결 → 되는 추적기(연속 심박·피부온도)를 켠다. 결과는 sink.state 와 info 로. */
    void start(boolean heart, boolean skinTemp) {
        try {
            Class<?> listenerC = Class.forName(PKG + ".ConnectionListener");
            Object listener = Proxy.newProxyInstance(listenerC.getClassLoader(), new Class<?>[]{listenerC}, (p, m, a) -> {
                switch (m.getName()) {
                    case "onConnectionSuccess": onConnected(heart, skinTemp); break;
                    case "onConnectionEnded": sink.state(Tr.s("samsung: 연결 끝", "samsung: ended")); break;
                    case "onConnectionFailed": failed(a != null && a.length > 0 ? a[0] : null); break;
                    default: return defaults(p, m, a);
                }
                return null;
            });
            service = Class.forName(PKG + ".HealthTrackingService").getConstructor(listenerC, Context.class).newInstance(listener, ctx);
            service.getClass().getMethod("connectService").invoke(service);
            info.put("sdk", true);
        } catch (Throwable t) {
            put("error", "connect: " + t);
            sink.state(Tr.s("samsung: 연결 못 함 (", "samsung: failed (") + t.getClass().getSimpleName() + ")");
        }
    }

    private void failed(Object e) {
        String why = String.valueOf(e);
        try {
            int code = (int) e.getClass().getMethod("getErrorCode").invoke(e);
            why = "error " + code + (Boolean.TRUE.equals(e.getClass().getMethod("hasResolution").invoke(e))
                    ? Tr.s(" (워치 Health Platform 개발자 모드 또는 권한이 필요)", " (needs Health Platform dev mode or permission)") : "");
        } catch (Throwable ignored) {
        }
        put("error", why);
        sink.state("samsung: " + why);
    }

    private void onConnected(boolean heart, boolean skinTemp) {
        try {
            Object cap = service.getClass().getMethod("getTrackingCapability").invoke(service);
            List<?> types = (List<?>) cap.getClass().getMethod("getSupportHealthTrackerTypes").invoke(cap);
            JSONArray names = new JSONArray();
            for (Object t : types) names.put(String.valueOf(t));
            info.put("supported", names);
            Class<?> typeC = Class.forName(PKG + ".data.HealthTrackerType");
            List<String> want = new ArrayList<>();
            if (heart) want.add("HEART_RATE_CONTINUOUS");
            if (skinTemp) { want.add("SKIN_TEMPERATURE_CONTINUOUS"); }
            JSONArray on = new JSONArray();
            for (String w : want) {
                Object type = null;
                for (Object t : types) if (String.valueOf(t).equals(w)) type = t;
                if (type == null) continue;
                Object tracker = service.getClass().getMethod("getHealthTracker", typeC).invoke(service, type);
                Class<?> evC = Class.forName(PKG + ".HealthTracker$TrackerEventListener");
                String kind = w;
                Object ev = Proxy.newProxyInstance(evC.getClassLoader(), new Class<?>[]{evC}, (p, m, a) -> {
                    if (m.getName().equals("onDataReceived")) data(kind, (List<?>) a[0]);
                    else if (m.getName().equals("onError")) sink.state("samsung " + kind + Tr.s(" 오류: ", " error: ") + a[0]);
                    else if (!m.getName().equals("onFlushCompleted")) return defaults(p, m, a);
                    return null;
                });
                tracker.getClass().getMethod("setEventListener", evC).invoke(tracker, ev);
                trackers.add(tracker);
                on.put(w);
            }
            info.put("tracking", on);
            sink.state(on.length() > 0 ? "samsung: " + on.join(", ").replace("\"", "") : Tr.s("samsung: 연결됨 (이 워치는 해당 추적 없음)", "samsung: on (no trackers)"));
        } catch (Throwable t) {
            put("error", "track: " + t);
            sink.state(Tr.s("samsung: 추적기 못 켬 (", "samsung: tracker failed (") + t.getClass().getSimpleName() + ")");
        }
    }

    private void data(String kind, List<?> points) {
        long bootMinusWall = SystemClock.elapsedRealtimeNanos() - System.currentTimeMillis() * 1_000_000L;
        for (Object dp : points) {
            try {
                long tMs = (long) dp.getClass().getMethod("getTimestamp").invoke(dp);
                long tNs = tMs * 1_000_000L + bootMinusWall;
                if (kind.startsWith("HEART_RATE")) {
                    Integer bpm = (Integer) value(dp, "HeartRateSet", "HEART_RATE");
                    Integer st = (Integer) value(dp, "HeartRateSet", "HEART_RATE_STATUS");
                    if (bpm != null) sink.heartRate(tNs, bpm, st == null ? -99 : st);
                    List<?> ibi = (List<?>) value(dp, "HeartRateSet", "IBI_LIST");
                    List<?> ibiSt = (List<?>) value(dp, "HeartRateSet", "IBI_STATUS_LIST");
                    for (int i = 0; ibi != null && i < ibi.size(); i++) {
                        int s = ibiSt != null && i < ibiSt.size() ? ((Number) ibiSt.get(i)).intValue() : -99;
                        sink.ibi(tNs, ((Number) ibi.get(i)).intValue(), s);
                    }
                } else {
                    Number obj = (Number) value(dp, "SkinTemperatureSet", "OBJECT_TEMPERATURE");
                    Number amb = (Number) value(dp, "SkinTemperatureSet", "AMBIENT_TEMPERATURE");
                    Number st = (Number) value(dp, "SkinTemperatureSet", "STATUS");      // SDK 1.4.1 에는 없다 → NO_STATUS
                    if (obj != null) sink.skin(tNs, obj.floatValue(), amb == null ? Float.NaN : amb.floatValue(), st == null ? NO_STATUS : st.intValue());
                }
            } catch (Throwable ignored) {
            }
        }
    }

    /** dp.getValue(ValueKey.<set>.<key>) — 키가 없는 SDK 버전이면 null. */
    private static Object value(Object dp, String set, String key) {
        try {
            Object k = Class.forName(PKG + ".data.ValueKey$" + set).getField(key).get(null);
            Class<?> vk = Class.forName(PKG + ".data.ValueKey");
            return dp.getClass().getMethod("getValue", vk).invoke(dp, k);
        } catch (Throwable t) {
            return null;
        }
    }

    void stop() {
        for (Object t : trackers) {
            try { t.getClass().getMethod("unsetEventListener").invoke(t); } catch (Throwable ignored) {}
        }
        trackers.clear();
        if (service != null) {
            try { service.getClass().getMethod("disconnectService").invoke(service); } catch (Throwable ignored) {}
        }
    }

    private void put(String k, String v) {
        try { info.put(k, v); } catch (Exception ignored) {}
    }

    /** 프록시의 Object 메서드 (equals·hashCode·toString). */
    private static Object defaults(Object proxy, Method m, Object[] a) {
        switch (m.getName()) {
            case "equals": return proxy == a[0];
            case "hashCode": return System.identityHashCode(proxy);
            case "toString": return "SamsungHealthProxy";
            default: return null;
        }
    }
}
