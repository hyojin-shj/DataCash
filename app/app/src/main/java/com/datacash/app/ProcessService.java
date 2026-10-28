package com.datacash.app;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.os.Build;
import android.os.IBinder;
import android.os.PowerManager;

import org.json.JSONObject;

import java.io.File;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * 녹화 처리 (앞에 떠 있는 서비스): 녹화를 하나씩 올리고 → 서버 검수를 20초마다 확인 (최대 30분).
 * 앱을 닫아도 계속된다. 진행 알림 '녹화를 처리하고 있어요', 끝나면 녹화마다 한 번 결과 알림
 * ('띠링! ○○ 미션 N원 적립' 또는 '검수 결과 · 이의 신청할 수 있어요').
 */
public class ProcessService extends Service {
    private static final String CH_WORK = "work", CH_REWARD = "reward";
    private static final int ID_WORK = 1;
    /** 올리기는 하나씩 (대역폭), 검수 기다리기는 녹화마다 따로 — 앞 녹화 검수 중에도 다음 녹화가 바로 올라간다. */
    private final ExecutorService one = Executors.newSingleThreadExecutor();
    private final ExecutorService wait = Executors.newCachedThreadPool();
    private int pending;
    private PowerManager.WakeLock lock;

    /** 이 프로세스에서 맡은 녹화 — 같은 녹화를 두 번 올리지 않게. */
    private static final java.util.Set<String> active = java.util.Collections.synchronizedSet(new java.util.HashSet<>());

    static void enqueue(Context c, File dir) {
        if (!active.add(dir.getAbsolutePath())) return;
        Intent i = new Intent(c, ProcessService.class).putExtra("dir", dir.getAbsolutePath());
        if (Build.VERSION.SDK_INT >= 26) c.startForegroundService(i); else c.startService(i);
    }

    @Override
    public void onCreate() {
        super.onCreate();
        NotificationManager nm = getSystemService(NotificationManager.class);
        nm.createNotificationChannel(new NotificationChannel(CH_WORK, "녹화 처리", NotificationManager.IMPORTANCE_LOW));
        nm.createNotificationChannel(new NotificationChannel(CH_REWARD, "보상 적립", NotificationManager.IMPORTANCE_HIGH));
        lock = ((PowerManager) getSystemService(POWER_SERVICE)).newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "datacash:process");
        lock.acquire(2 * 60 * 60 * 1000L);
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        Notification n = note(CH_WORK, "녹화를 처리하고 있어요", "앱을 닫아도 계속돼요", true);
        if (Build.VERSION.SDK_INT >= 29) startForeground(ID_WORK, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC);
        else startForeground(ID_WORK, n);
        String path = intent == null ? null : intent.getStringExtra("dir");
        if (path != null) {
            synchronized (this) { pending++; }
            one.execute(() -> {
                File dir = new File(path);
                if (upload(dir)) wait.execute(() -> { await(dir); active.remove(path); finished(); });
                else { active.remove(path); finished(); }
            });
        }
        return START_REDELIVER_INTENT;
    }

    /** 앱이 꺼지거나 새로 설치돼 끊긴 녹화(저장됨·올리는 중)를 다시 맡는다. */
    static void resume(Context c) {
        for (File d : Store.sessions(c)) {
            String s = Uploader.localState(d).optString("state");
            if (s.equals("saved") || s.equals("uploading")) enqueue(c, d);
        }
    }

    private synchronized void finished() {
        if (--pending == 0) { stopForeground(STOP_FOREGROUND_REMOVE); stopSelf(); }
    }

    private boolean upload(File dir) {
        String title = Uploader.localState(dir).optString("title", "녹화");
        try {
            if (!Uploader.localState(dir).has("state") || "upload_failed".equals(Uploader.localState(dir).optString("state"))
                    || "saved".equals(Uploader.localState(dir).optString("state"))
                    || "uploading".equals(Uploader.localState(dir).optString("state"))) {
                Discover.ensure(this);
                Uploader.upload(this, dir, (step, pct) -> progress(title + " 녹화 · 올리는 중 " + pct + "%", "서버에서 얼굴을 바꾸고 가린 뒤 원본은 바로 지워요"));
            }
            return true;
        } catch (Exception e) {
            Uploader.save(dir, "state", "upload_failed");
            note(CH_REWARD, title + " 녹화를 올리지 못했어요", "내 녹화 기록에서 '다시 올리기'를 눌러 주세요", false, dir.getName().hashCode());
            return false;
        }
    }

    private void await(File dir) {
        String title = Uploader.localState(dir).optString("title", "녹화");
        try {
            long t0 = System.currentTimeMillis();
            while (System.currentTimeMillis() - t0 < 30 * 60_000L) {
                JSONObject st = Uploader.status(this, dir);
                String s = st.optString("state");
                boolean paid = st.optJSONObject("payout") != null;
                if (s.equals("done") && paid || s.equals("rejected") || s.equals("failed")) { result(dir, title, st); return; }
                progress(title + " 녹화 · 검수 중", "가리기 · 라벨 · 4감 맞추기 — 끝나면 알려 드릴게요");
                Thread.sleep(20_000);
            }
        } catch (Exception ignored) {
            // 상태를 못 읽어도 녹화는 이미 올라갔다 — 내 녹화 기록에서 다시 확인된다
        }
    }

    private void progress(String title, String text) {
        getSystemService(NotificationManager.class).notify(ID_WORK, note(CH_WORK, title, text, true));
    }

    /** 결과 알림은 녹화마다 한 번. 서버 쪽 실패는 알리지 않는다 (내 녹화 기록에 보인다). */
    private void result(File dir, String title, JSONObject st) {
        if (Uploader.localState(dir).optBoolean("notified")) return;
        Uploader.save(dir, "notified", true);
        String s = st.optString("state");
        if (s.equals("done")) {
            JSONObject py = st.optJSONObject("payout");
            int won = py != null ? py.optInt("amount") : st.optInt("reward");
            boolean sent = py != null && py.optString("status").equals("sent");
            note(CH_REWARD, String.format(Locale.KOREA, "띠링! %s 미션 %,d원 적립", title, won),
                    sent ? py.optString("label") + " 으로 입금됐어요" : "계좌를 연결하면 바로 입금돼요", false, dir.getName().hashCode());
        } else if (s.equals("rejected")) {
            note(CH_REWARD, title + " 미션 검수 결과", st.optString("error", "반려됐어요") + " · 이의 신청할 수 있어요", false, dir.getName().hashCode());
        }
    }

    private Notification note(String ch, String title, String text, boolean ongoing) {
        PendingIntent open = PendingIntent.getActivity(this, 0, new Intent(this, HomeActivity.class)
                .addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_SINGLE_TOP), PendingIntent.FLAG_IMMUTABLE);
        return new Notification.Builder(this, ch).setSmallIcon(R.drawable.ic_note).setContentTitle(title).setContentText(text)
                .setStyle(new Notification.BigTextStyle().bigText(text)).setOngoing(ongoing).setAutoCancel(!ongoing)
                .setContentIntent(open).build();
    }

    private void note(String ch, String title, String text, boolean ongoing, int id) {
        getSystemService(NotificationManager.class).notify(1000 + (id & 0xFFFFF), note(ch, title, text, ongoing));
    }

    @Override
    public void onDestroy() {
        if (lock != null && lock.isHeld()) lock.release();
        one.shutdownNow();
        super.onDestroy();
    }

    @Override
    public IBinder onBind(Intent i) { return null; }
}
