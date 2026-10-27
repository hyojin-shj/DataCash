# 갤럭시 워치 심박·피부온도 — Samsung Health Sensor SDK 붙이기

데이터캐시 워치 앱(`app/wear`)은 SDK 없이도 돈다. SDK 를 넣으면 더 많은 값을 기록한다.

| 값 | SDK 없을 때 | SDK 있을 때 |
|---|---|---|
| 진동 펄스 (촉각·동기) | 안드로이드 기본 진동 | 같음 |
| 심박 | 안드로이드 기본 심박 센서 (`TYPE_HEART_RATE`) | 연속 심박 `HEART_RATE_CONTINUOUS` + 측정 상태 |
| 박동 간격 (IBI) | 없음 | `watch_ibi.csv` — 심박 변이도(HRV)에 쓸 수 있다 |
| 피부온도 | 없음 (공개 센서로 안 나옴) | `SKIN_TEMPERATURE_CONTINUOUS` → `watch_skin.csv` (주변 온도 같이) |

생체 값은 **생체 신호 동의(본인)** 를 한 녹화에서만 기록한다.

## 1. SDK 받기 (직접)

1. [Samsung Developers](https://developer.samsung.com/health/sensor/overview.html) 의 **Samsung Health Sensor SDK** 에서 받는다.
   삼성 계정 로그인과 이용 약관 동의가 필요하다.
2. 받은 묶음 안의 `samsung-health-sensor-api-*.aar` 를 `app/wear/libs/` 에 넣는다.
   - 파일 이름은 그대로 둬도 된다. `libs/` 안의 모든 `.aar` 를 같이 넣는다.
   - 삼성 약관상 저장소에는 올리지 않는다. `.gitignore` 에 이미 넣어 두었다.

## 2. 워치 준비 (파트너 등록 전 시험용)

워치에서 **Health Platform 을 개발자 모드로** 켠다.

1. **설정 → 애플리케이션 → Health Platform** 을 연다.
2. 제목(Health Platform)을 여러 번(10번쯤) 누르면 `[Dev mode]` 가 표시된다.

정식 배포에는 삼성 파트너 등록이 필요하다.
SDK 문서의 **지원 기기·버전 표**를 확인한다. 피부온도 연속 측정은 지원 워치와 SDK 버전에서만 나온다.

## 3. 빌드·설치

```bash
cd app && JAVA_HOME=~/Library/Android/jdk/Contents/Home ./gradlew :wear:assembleDebug
```

워치를 무선 디버깅으로 연결한 뒤 설치한다.

```bash
adb -s <워치> install -r -g wear/build/outputs/apk/debug/wear-debug.apk
```

## 4. 확인

1. 폰 앱에서 생체 신호 동의를 한 상태로 녹화를 시작한다.
2. 워치 화면에 `● 손목 기록 0:05` 아래로 다음과 같이 나오면 된다.
   - `samsung: HEART_RATE_CONTINUOUS, SKIN_TEMPERATURE_CONTINUOUS` — SDK 연결됨
   - `samsung: error … (워치 Health Platform 개발자 모드 또는 권한이 필요)` — 2단계를 확인한다. 이때 앱은 저절로 기본 심박 센서로 바꾼다.
   - `기본 심박 센서 (피부온도는 삼성 SDK 필요)` — SDK 파일이 앱에 안 들어갔다 (1단계).
3. 올린 뒤 서버의 `watch_meta.json` 에서 연결 결과를 볼 수 있다.
   - `samsung.supported`: 이 워치가 지원하는 추적 종류
   - `samsung.tracking`: 켠 추적
   - `samsung.error`: 오류

## 구현 메모

- **부르는 방식:** `app/wear/src/main/java/com/datacache/wear/SamsungHealth.java` — SDK 클래스를 리플렉션으로 부른다.
  그래서 SDK 가 없어도 컴파일되고, 있으면 저절로 쓴다.
  - 연결: `HealthTrackingService.connectService()`
  - 지원 종류 확인: `getTrackingCapability().getSupportHealthTrackerTypes()`
  - 추적기 켜기: `getHealthTracker(type)` → `setEventListener`
- **시각:** SDK 는 벽시계(ms)를 준다. 받은 순간의 (BOOTTIME − 벽시계) 차이로 BOOTTIME ns 로 옮겨, 손목 IMU·서버 시계 맞춤과 같은 시계로 둔다.
- **기록 조건 (성공한 측정만):**
  - 심박: 상태 1(측정 성공)
  - 박동 간격: 상태 0(정상)
  - 피부온도: 상태 0(측정 성공). SDK 1.4.1 은 피부온도에 상태 값이 없다 → 사람 피부 범위(20~45°C)만, 상태 칸은 -1
- **SDK 1.4.1 과 대조함 (10-27):** 클래스·메서드·값 이름이 모두 맞다 (피부온도 STATUS 만 없음). 실기기 확인은 워치가 오는 10-29.
  SDK 버전마다 이름(`ValueKey` 등)이 다르면, 그 값만 빠지고 나머지는 그대로 기록된다.
