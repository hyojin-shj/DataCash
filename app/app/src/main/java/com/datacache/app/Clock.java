package com.datacache.app;

import android.os.SystemClock;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

/** 모든 기록은 폰의 부팅 후 시계(BOOTTIME, 잠들어도 흐름)로. 카메라 SENSOR_TIMESTAMP(REALTIME)·센서·AudioTimestamp 와 같은 기준. */
final class Clock {
    private Clock() {}

    static long now() {
        return SystemClock.elapsedRealtimeNanos();
    }

    static String wallIso() {
        return new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ssZ", Locale.US).format(new Date());
    }

    static String sessionId() {
        return new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
    }
}
