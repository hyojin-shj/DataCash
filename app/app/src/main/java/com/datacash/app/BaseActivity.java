package com.datacash.app;

import android.app.Activity;
import android.content.Context;
import android.os.Bundle;

/** 모든 화면의 바탕: 고른 언어로 감싸고, 다른 화면에서 언어를 바꾸고 돌아오면 다시 만든다. */
public class BaseActivity extends Activity {
    private String lang;

    @Override
    protected void attachBaseContext(Context base) {
        super.attachBaseContext(Lang.wrap(base));
    }

    @Override
    protected void onCreate(Bundle state) {
        lang = Store.lang(this);
        super.onCreate(state);
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (lang != null && !lang.equals(Store.lang(this))) recreate();
    }
}
