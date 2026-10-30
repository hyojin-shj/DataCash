package com.datacash.app;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.text.SpannableString;
import android.text.style.ForegroundColorSpan;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.View;
import android.widget.FrameLayout;
import android.widget.HorizontalScrollView;
import android.widget.LinearLayout;
import android.widget.TextView;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * 첫 화면.
 *   모은 보상        크게 + 사물에 빗댄 한 줄 ("치킨 한 마리 값을 벌었어요!"), 다음 목표는 '미션 몇 개'로
 *   오늘의 보너스 미션 하루에 하나 +50% (서버와 같은 식으로 고름). 진행 고리 가운데 빨간 버튼 → 아직 안 찍은 다음 상황 촬영
 *   기업 의뢰 미션    옆으로 넘기는 카드: 보상 · 의뢰 목적 · 찍은 상황 수 · 상황 골라서 찍기
 *   더 많은 기능      보상 받을 계좌 · 스마트 기기 연결 · 쥐는 세기 맞추기 · 내 녹화 기록 · 동의 다시 보기
 * 처음이면 동의 화면부터.
 */
public class HomeActivity extends BaseActivity {
    private final Map<String, Integer> scarce = new HashMap<>();   // 서버가 알려 주는 부족한 미션 보너스 %
    private String todayId;
    private LinearLayout pages, dots;
    private HorizontalScrollView pager;

    private final android.os.Handler ui = new android.os.Handler(android.os.Looper.getMainLooper());
    private boolean visible;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        if (!Store.consented(this)) startActivity(new Intent(this, ConsentActivity.class));
        if (android.os.Build.VERSION.SDK_INT >= 33
                && checkSelfPermission(android.Manifest.permission.POST_NOTIFICATIONS) != android.content.pm.PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{android.Manifest.permission.POST_NOTIFICATIONS}, 11);   // 적립 알림
        }
        for (File d : Store.sessions(this)) {     // 앱이 꺼져서 멈춘 올리기를 다시 맡긴다
            String st = Uploader.localState(d).optString("state");
            if ((st.equals("uploading") || st.equals("saved")) && new File(d, "meta.json").exists()) {
                Uploader.save(d, "state", "saved");
                ProcessService.enqueue(this, d);
            }
        }
    }

    /** 떠 있는 동안 5초마다 처리 중인 녹화의 결과를 받고, 입금되면 '띠링'. */
    private void poll() {
        if (!visible) return;
        new Thread(() -> {
            String before = Uploader.signature(this);
            Uploader.refreshPending(this);
            boolean changed = !before.equals(Uploader.signature(this));
            ui.post(() -> {
                if (!visible) return;
                if (changed) render();          // 바뀐 게 있을 때만 다시 그린다 (스크롤이 튀지 않게)
                Payout.checkNew(this);
                ui.postDelayed(this::poll, 5000);
            });
        }).start();
    }

    @Override
    protected void onPause() {
        super.onPause();
        visible = false;
    }

    @Override
    protected void onResume() {
        super.onResume();
        visible = true;
        ProcessService.resume(this);
        ui.postDelayed(this::poll, 1500);
        render();
        new Thread(() -> {
            try {
                Discover.ensure(this);           // 맥 IP 가 바뀌었으면 서버를 다시 찾는다
                JSONArray ms = Uploader.missions(this);
                Map<String, Integer> b = new HashMap<>();
                String t = null;
                for (int i = 0; i < ms.length(); i++) {
                    JSONObject m = ms.getJSONObject(i);
                    if (m.optInt("bonus_pct") > 0) b.put(m.getString("id"), m.getInt("bonus_pct"));
                    if (m.optBoolean("today")) t = m.getString("id");
                }
                String today = t;
                boolean fresh = Uploader.changed;
                runOnUiThread(() -> {
                    boolean same = scarce.equals(b) && (today == null || today.equals(todayId)) && !fresh;
                    scarce.clear();
                    scarce.putAll(b);
                    if (today != null) todayId = today;
                    if (!same) render();          // 의뢰 미션이 새로 열렸거나 보너스가 바뀌었을 때만 다시 그린다
                });
            } catch (Exception ignored) {   // 서버가 없으면 폰 안의 미션 목록만으로
            }
        }).start();
    }

    private void render() {
        int todayCount = 0, todayWon = 0;
        String todayPrefix = Clock.sessionId().substring(0, 8);
        for (File d : Store.sessions(this)) {
            if (!d.getName().startsWith(todayPrefix)) continue;
            todayCount++;
            JSONObject st = Uploader.localState(d);
            if (st.optString("state").equals("done")) todayWon += st.optInt("reward") + st.optInt("appeal_extra");
        }
        TextView badge = Ui.chip(this, todayWon > 0 ? getString(R.string.today_won, todayWon)
                : todayCount == 0 ? getString(R.string.today_first) : getString(R.string.today_count, todayCount), Ui.PRIMARY, Ui.PRIMARY_SOFT);
        badge.setTextSize(13);
        LinearLayout right = new LinearLayout(this);       // 언어 고르기 + 오늘 뱃지
        right.setGravity(Gravity.CENTER_VERTICAL);
        right.addView(Lang.button(this), Ui.margins(this, -2, -2, 0, 0, 0, 6, 0));
        right.addView(badge);
        LinearLayout body = Ui.page(this, null, false, right);
        try {
            if (todayId == null) todayId = Store.todayMission(this);
            Ui.add(this, body, earnings());
            Ui.add(this, body, todayCard(Store.mission(this, todayId)));
            LinearLayout sec = new LinearLayout(this);
            sec.setGravity(Gravity.CENTER_VERTICAL);
            sec.addView(Ui.sectionTitle(this, getString(R.string.home_sponsored)), Ui.weight());
            TextView hint = Ui.text(this, getString(R.string.swipe_hint), 13, Ui.SUB, false);
            hint.setPadding(0, Ui.dp(this, 12), Ui.dp(this, 4), Ui.dp(this, 10));
            sec.addView(hint);
            body.addView(sec);
            body.addView(pager());
            dots = new LinearLayout(this);
            dots.setGravity(Gravity.CENTER);
            body.addView(dots, Ui.margins(this, -1, -2, 0, 0, 12, 0, 6));
            showDots(0);
        } catch (Exception e) {
            body.addView(Ui.text(this, getString(R.string.missions_read_fail, e.getMessage()), 13, Ui.SUB, false));
        }

        body.addView(Ui.sectionTitle(this, getString(R.string.more_features)));
        LinearLayout menu = Ui.listCard(this);
        menu.addView(Ui.row(this, Icon.STAR, Ui.ACCENT_INK, Ui.ACCENT_SOFT, getString(R.string.menu_payout),
                Store.account(this).isEmpty() ? getString(R.string.menu_payout_desc) : PayoutActivity.bankLabel(this, Store.bank(this)) + " " + Store.maskedAccount(this),
                null, v -> open(PayoutActivity.class)));
        menu.addView(Ui.row(this, Icon.WATCH, Ui.PRIMARY, Ui.PRIMARY_SOFT, getString(R.string.menu_devices),
                getString(R.string.menu_devices_desc), null, v -> open(DevicesActivity.class)));
        menu.addView(Ui.row(this, Icon.HAND, Ui.PRIMARY, Ui.PRIMARY_SOFT, getString(R.string.menu_calib),
                getString(R.string.menu_calib_desc), null, v -> open(CalibrationActivity.class)));
        menu.addView(Ui.row(this, Icon.PERSON, Ui.OK, Ui.OK_SOFT, getString(R.string.menu_people),
                getString(R.string.menu_people_desc), null, v -> open(PeopleActivity.class)));
        menu.addView(Ui.row(this, Icon.FILM, Ui.ACCENT_INK, Ui.ACCENT_SOFT, getString(R.string.menu_history),
                getString(R.string.menu_history_desc), null, v -> open(HistoryActivity.class)));
        menu.addView(Ui.row(this, Icon.LIST, Ui.VIOLET, Ui.VIOLET_SOFT, getString(R.string.menu_appeals),
                getString(R.string.menu_appeals_desc), null, v -> open(AppealsActivity.class)));
        Ui.add(this, body, menu);

        LinearLayout privacy = Ui.listCard(this);
        privacy.addView(Ui.row(this, Icon.SHIELD, Ui.SUB, Ui.FIELD, getString(R.string.privacy_title),
                getString(R.string.privacy_desc), null, null));
        Ui.add(this, body, privacy);
    }

    private void open(Class<?> c) {
        startActivity(new Intent(this, c));
    }

    // ---------- 모은 보상 ----------

    /** 번 돈 이하인 가장 비싼 것에 빗댄다 (2026년 대략 가격). 숫자보다 사물이 먼저 와닿게. */
    private static final int[] PRICE = {1_500, 4_500, 10_000, 15_000, 22_000, 55_000, 120_000, 300_000, 600_000};

    private LinearLayout earnings() {
        int won = Store.earned(this);
        String[] THING = getResources().getStringArray(R.array.things);   // PRICE 와 같은 순서
        LinearLayout c = Ui.card(this);
        c.setPadding(Ui.dp(this, 22), Ui.dp(this, 20), Ui.dp(this, 22), Ui.dp(this, 20));
        c.addView(Ui.text(this, getString(R.string.earned_title), 14, Ui.SUB, true));
        TextView big = Ui.text(this, Ui.won(this, won), 34, Ui.INK, true);
        c.addView(big, Ui.margins(this, -2, -2, 0, 0, 6, 0, 0));
        int i = -1;
        while (i + 1 < PRICE.length && won >= PRICE[i + 1]) i++;
        TextView like = Ui.text(this, "", 17, Ui.INK, true);
        if (i < 0) {
            like.setText(getString(R.string.earned_first));
        } else {
            String full = getString(R.string.earned_like, THING[i]);
            SpannableString s = new SpannableString(full);
            int at = Math.max(0, full.indexOf(THING[i]));
            s.setSpan(new ForegroundColorSpan(Ui.ACCENT_INK), at, Math.min(full.length(), at + THING[i].length()), 0);
            like.setText(s);
        }
        like.setPadding(0, Ui.dp(this, 8), 0, 0);
        c.addView(like);
        if (i + 1 < PRICE.length) {
            int per = Store.perMission(this);
            int n = (PRICE[i + 1] - won + per - 1) / per;
            TextView sub = Ui.text(this, getString(R.string.earned_next, n, THING[i + 1]), 13, Ui.SUB, false);
            sub.setPadding(0, Ui.dp(this, 8), 0, 0);
            c.addView(sub);
        }
        c.setOnClickListener(v -> open(PayoutActivity.class));
        return c;
    }

    // ---------- 오늘의 보너스 미션 ----------

    private LinearLayout todayCard(JSONObject m) throws Exception {
        String id = m.getString("id");
        JSONArray sits = m.getJSONArray("situations");
        Set<String> done = Store.doneSituations(this, id);
        LinearLayout c = Ui.card(this);
        c.setGravity(Gravity.CENTER_HORIZONTAL);
        c.setPadding(Ui.dp(this, 20), Ui.dp(this, 22), Ui.dp(this, 20), Ui.dp(this, 20));
        LinearLayout badge = new LinearLayout(this);
        badge.setGravity(Gravity.CENTER_VERTICAL);
        badge.setBackground(Ui.round(Ui.ACCENT_SOFT, Ui.dp(this, 20)));
        badge.setPadding(Ui.dp(this, 10), Ui.dp(this, 5), Ui.dp(this, 12), Ui.dp(this, 5));
        badge.addView(new Icon(this, Icon.STAR, Ui.ACCENT_INK), new LinearLayout.LayoutParams(Ui.dp(this, 14), Ui.dp(this, 14)));
        TextView bt = Ui.text(this, getString(R.string.today_badge), 12, Ui.ACCENT_INK, true);
        bt.setPadding(Ui.dp(this, 5), 0, 0, 0);
        badge.addView(bt);
        c.addView(badge, Ui.wrap());
        TextView title = Ui.text(this, m.getString("title"), 30, Ui.INK, true);
        title.setPadding(0, Ui.dp(this, 12), 0, Ui.dp(this, 6));
        c.addView(title, Ui.wrap());
        TextView guide = Ui.text(this, m.getString("guide"), 14, Ui.SUB, false);
        guide.setGravity(Gravity.CENTER);
        c.addView(guide, Ui.wrap());

        RecordRing ring = new RecordRing(this);
        ring.setProgress(sits.length() == 0 ? 0f : (float) done.size() / sits.length());
        ring.setContentDescription(getString(R.string.start_recording));
        int next = 0;
        while (next < sits.length() && done.contains(sits.getString(next))) next++;
        int nextIdx = next >= sits.length() ? 0 : next;
        ring.setOnClickListener(v -> record(id, sits.optString(nextIdx)));
        c.addView(ring, Ui.margins(this, Ui.dp(this, 200), Ui.dp(this, 200), 0, 0, 16, 0, 12));

        int base = Store.perMission(this), bonus = base / 2;
        LinearLayout rw = new LinearLayout(this);
        rw.setGravity(Gravity.CENTER_VERTICAL);
        rw.addView(Ui.text(this, Ui.plusWon(this, base + bonus), 26, Ui.PRIMARY, true));
        rw.addView(Ui.text(this, "  " + getString(R.string.reward_breakdown, base, bonus), 13, Ui.SUB, false));
        c.addView(rw, Ui.wrap());
        TextView cnt = Ui.text(this, getString(R.string.tap_next, nextIdx + 1, sits.length()),
                13, Ui.SUB, false);
        cnt.setPadding(0, Ui.dp(this, 8), 0, 0);
        c.addView(cnt, Ui.wrap());
        TextView pick = Ui.text(this, getString(R.string.pick_situation), 14, Ui.PRIMARY, true);
        pick.setPadding(Ui.dp(this, 12), Ui.dp(this, 12), Ui.dp(this, 12), Ui.dp(this, 2));
        pick.setBackground(Ui.pressable(android.graphics.Color.TRANSPARENT, Ui.dp(this, 12)));
        pick.setOnClickListener(v -> detail(id));
        c.addView(pick, Ui.wrap());
        return c;
    }

    private void record(String missionId, String situation) {
        startActivity(new Intent(this, RecordActivity.class).putExtra("mission", missionId).putExtra("situation", situation));
    }

    private void detail(String missionId) {
        startActivity(new Intent(this, MissionActivity.class).putExtra("mission", missionId));
    }

    // ---------- 옆으로 넘기는 미션 카드 ----------

    private int pageWidth() {
        return getResources().getDisplayMetrics().widthPixels - Ui.dp(this, 32) - Ui.dp(this, 28);   // 다음 장이 조금 보이게
    }

    private View pager() throws Exception {
        pager = new HorizontalScrollView(this) {
            private float downX, downY;
            private int downScroll;

            @Override
            public boolean onInterceptTouchEvent(MotionEvent e) {
                if (e.getAction() == MotionEvent.ACTION_DOWN) { downX = e.getX(); downY = e.getY(); downScroll = getScrollX(); }
                else if (e.getAction() == MotionEvent.ACTION_MOVE
                        && Math.abs(e.getX() - downX) > Math.abs(e.getY() - downY)) getParent().requestDisallowInterceptTouchEvent(true);  // 옆으로 밀면 위아래 스크롤이 가로채지 않게
                return super.onInterceptTouchEvent(e);
            }

            @Override
            public boolean onTouchEvent(MotionEvent e) {
                if (e.getAction() == MotionEvent.ACTION_DOWN) { downX = e.getX(); downY = e.getY(); downScroll = getScrollX(); }
                boolean r = super.onTouchEvent(e);
                if (e.getAction() == MotionEvent.ACTION_UP || e.getAction() == MotionEvent.ACTION_CANCEL) {
                    int moved = getScrollX() - downScroll;
                    post(() -> snapFrom(downScroll, moved));
                }
                return r;
            }

            @Override
            public void fling(int vx) {}   // 손을 뗄 때 snapFrom 이 정한다 (여기서 움직이면 snap 과 겹쳐 제자리로 돌아갔다)
        };
        pager.setHorizontalScrollBarEnabled(false);
        pager.setClipToPadding(false);
        pages = new LinearLayout(this);
        JSONArray ms = Store.missions(this);
        for (int i = 0; i < ms.length(); i++) {
            JSONObject m = ms.getJSONObject(i);
            pages.addView(missionCard(m), Ui.margins(this, pageWidth(), -2, 0, 0, 0, 12, 0));
        }
        pages.addView(summaryCard(), Ui.margins(this, pageWidth(), -1, 0, 0, 0, 12, 0));
        pager.addView(pages);
        return pager;
    }

    /** 손을 뗐을 때: 처음 장에서 조금(장 너비의 12% 또는 32dp)만 밀어도 민 쪽으로 한 장. */
    private void snapFrom(int startScroll, int moved) {
        int step = pageWidth() + Ui.dp(this, 12);
        int cur = Math.round((float) startScroll / step);
        int need = Math.min(Ui.dp(this, 32), step * 12 / 100);
        int to = Math.abs(moved) >= need ? cur + (moved > 0 ? 1 : -1) : cur;
        to = Math.max(0, Math.min(pages.getChildCount() - 1, to));
        pager.smoothScrollTo(to * step, 0);
        showDots(to);
    }

    private void showDots(int sel) {
        if (dots == null || pages == null) return;
        dots.removeAllViews();
        for (int i = 0; i < pages.getChildCount(); i++) {
            View d = new View(this);
            d.setBackground(Ui.round(i == sel ? Ui.PRIMARY : Ui.LINE, Ui.dp(this, 4)));
            dots.addView(d, Ui.margins(this, Ui.dp(this, i == sel ? 18 : 7), Ui.dp(this, 7), 0, 3, 0, 3, 0));
        }
    }

    /** 마지막 장: 오늘 요약 (녹화 · 적립 예정 · 동의 인원). */
    private LinearLayout summaryCard() {
        int rec = 0, expect = 0;
        String todayPrefix = Clock.sessionId().substring(0, 8);
        for (File d : Store.sessions(this)) {
            if (!d.getName().startsWith(todayPrefix) || !new File(d, "meta.json").exists()) continue;
            rec++;
            JSONObject st = Uploader.localState(d);
            String s = st.optString("state");
            expect += s.equals("done") ? st.optInt("reward") + st.optInt("appeal_extra")
                    : s.equals("rejected") || s.equals("failed") ? 0 : Store.perMission(this);
        }
        LinearLayout c = Ui.card(this);
        FrameLayout tile = Ui.iconTile(this, Icon.SPARK, Ui.ACCENT_INK, Ui.ACCENT_SOFT, 46);
        c.addView(tile, new LinearLayout.LayoutParams(Ui.dp(this, 46), Ui.dp(this, 46)));
        TextView t = Ui.text(this, getString(R.string.summary_title), 21, Ui.INK, true);
        t.setPadding(0, Ui.dp(this, 14), 0, Ui.dp(this, 10));
        c.addView(t);
        String[][] rows = {{getString(R.string.sum_rec), getString(R.string.count_items, rec)}, {getString(R.string.sum_expect), Ui.won(this, expect)},
                {getString(R.string.sum_people), getString(Store.ownerFace(this) ? R.string.face_registered_me : R.string.face_needed_me)}};
        for (String[] r : rows) {
            LinearLayout line = new LinearLayout(this);
            line.setPadding(0, Ui.dp(this, 5), 0, Ui.dp(this, 5));
            line.addView(Ui.text(this, r[0], 14, Ui.SUB, false), Ui.weight());
            line.addView(Ui.text(this, r[1], 15, Ui.INK, true));
            c.addView(line);
        }
        TextView back = Ui.text(this, getString(R.string.summary_back), 12, Ui.SUB, false);
        back.setPadding(0, Ui.dp(this, 14), 0, 0);
        c.addView(back);
        return c;
    }

    private LinearLayout missionCard(JSONObject m) throws Exception {
        String id = m.getString("id");
        int total = m.getJSONArray("situations").length();
        int done = Store.doneSituations(this, id).size();
        LinearLayout c = Ui.card(this);
        LinearLayout top = new LinearLayout(this);
        top.setGravity(Gravity.CENTER_VERTICAL);
        FrameLayout tile = Ui.iconTile(this, Icon.TARGET, Ui.PRIMARY, Ui.PRIMARY_SOFT, 46);
        top.addView(tile, new LinearLayout.LayoutParams(Ui.dp(this, 46), Ui.dp(this, 46)));
        top.addView(new View(this), Ui.weight());
        int per = Store.perMission(this);
        boolean today = id.equals(todayId);
        Integer sc = scarce.get(id);
        int pct = (today ? 50 : 0) + (sc != null ? sc : 0);
        String label = Ui.plusWon(this, per * (100 + pct) / 100)
                + (today ? " · " + getString(R.string.today_bonus) : sc != null ? " · " + getString(R.string.scarce, sc) : "");
        top.addView(Ui.chip(this, label, Ui.ACCENT_INK, Ui.ACCENT_SOFT));
        c.addView(top);
        TextView t = Ui.text(this, m.getString("title"), 21, Ui.INK, true);
        t.setPadding(0, Ui.dp(this, 14), 0, Ui.dp(this, 6));
        c.addView(t);
        String sponsor = m.optString("sponsor");
        TextView p = Ui.text(this, sponsor.isEmpty() ? getString(R.string.req_purpose, m.getString("purpose"))
                : getString(R.string.req_sponsor_purpose, sponsor, m.getString("purpose")), 13, Ui.SUB, false);
        p.setMaxLines(2);
        p.setMinLines(2);
        c.addView(p);
        TextView n = Ui.text(this, getString(R.string.situations_count, done, total), 13, Ui.INK, true);
        n.setPadding(0, Ui.dp(this, 14), 0, Ui.dp(this, 8));
        c.addView(n);
        c.addView(Ui.progress(this, total == 0 ? 0f : (float) done / total), new LinearLayout.LayoutParams(-1, Ui.dp(this, 6)));
        c.addView(Ui.button(this, getString(R.string.pick_situation), true, v -> detail(id)), Ui.margins(this, -1, -2, 0, 0, 16, 0, 0));
        return c;
    }
}
