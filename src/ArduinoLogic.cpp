/**
 * 로직:
 *   - 복도 양쪽에 택배 보관 장소(스테이션)가 등간격으로 배치됨
 *   - 초음파로 "가까워짐→중앙 통과→멀어짐" 패턴을 1개 스테이션으로 카운트
 *   - 목표 스테이션에 도달하면 스테이션의 중심 옆에 정렬하여 정지
 *
 * 통신/프로토콜:
 *   EX)
 *   "R3\n" : 오른쪽 3번째 스테이션
 *   "L2\n" : 왼쪽  2번째 스테이션
 *
 * 센서:
 *   Rf = Right-Front  (오른앞 초음파)  [오른쪽 카운트/정렬]
 *   Rr = Right-Rear   (오른뒤  초음파) [오른쪽 카운트/정렬]
 *   F  = Front        (정면   초음파)  [안전/충돌 방지]
 *   Lf = Left-Front   (왼앞   초음파) // 있으면 추가
 *
 *
 * 안전:
 *   
 *
 * 참고:
 *   - F(정면) < 0.15m → 즉시 정지
 */

#include <Arduino.h>

// ===== Config toggles =====
#define HAS_LF 1   // 좌측 센서를 달았다면 1, 없으면 0 (왼쪽 정밀도에 큰 차이)

#if HAS_LF
#define TRIG_LF  A0
#define ECHO_LF  A1
#endif
#define TRIG_RF  2
#define ECHO_RF  3
#define TRIG_RR  4
#define ECHO_RR  5
#define TRIG_F   6
#define ECHO_F   7

#define L_PWM  9
#define L_IN1  8
#define L_IN2  10
#define R_PWM  11
#define R_IN1  12
#define R_IN2  13

// 서보모터 5, 6번핀

const float d_set_right = 0.28; // 오른쪽 벽/물체로부터 "보통 주행" 시 유지하고 싶은 거리
const float th_obj      = 0.10; // 스테이션 판정 임계(가까워짐). near = (distance < d_set_right - th_obj)
const float d_near      = d_set_right - th_obj;

const float d_stop_front = 0.25; // 전방 안전 정지 거리

const float Kp = 1.0;  // 오른쪽 거리 유지 게인 (d_set_right - Rf)
const float Ky = 1.2;  // yaw 보정 게인 (Rr - Rf)  (오른쪽 벽과 평행 정렬)

const int   N_frames = 4;    // 연속 프레임 유지 개수(디바운스)
const int   LOOP_HZ  = 20;
const int   LOOP_DT_MS = 1000/LOOP_HZ;

unsigned long T_back_ms   = 800;   // 드롭 후 약간 후진

const float V_FWD   = 0.8;  // 평속
const float V_ALIGN = 0.4;  // 정렬용 저속
const int PWM_BASE_FWD  = 160;
const int PWM_BASE_SLOW = 120;
const int PWM_MIN = 60;
const int PWM_MAX = 255;

// ===== State machine =====
enum Side { SIDE_R=0, SIDE_L=1 };
enum State { IDLE, FOLLOW_MAIN, ALIGN_AT_STATION, DROP, EXIT_FROM_STATION, DONE };
State state = IDLE;
Side  targetSide = SIDE_R;
int   targetIdx  = -1;   // 0..N
int   currentIdx = 0;

unsigned long lastLoop = 0;

float ema(float prev, float x, float a=0.35){ return prev<1e-6 ? x : a*x + (1-a)*prev; }
float emaLf=0, emaRf=0, emaRr=0, emaF=0;

float readUS(int trig, int echo){
  digitalWrite(trig, LOW); delayMicroseconds(2);
  digitalWrite(trig, HIGH); delayMicroseconds(10);
  digitalWrite(trig, LOW);
  unsigned long dur = pulseIn(echo, HIGH, 30000UL);
  if (dur==0) return 4.0; // timeout
  return (dur * 1e-6f) * 343.0f / 2.0f;
}

void setMotorLR(float cmdL, float cmdR, bool slow=false){
  int base = slow ? PWM_BASE_SLOW : PWM_BASE_FWD;
  cmdL = constrain(cmdL, -1.0, 1.0);
  cmdR = constrain(cmdR, -1.0, 1.0);
  int pwmL = base * fabs(cmdL);
  int pwmR = base * fabs(cmdR);
  pwmL = constrain(pwmL, 0, PWM_MAX);
  pwmR = constrain(pwmR, 0, PWM_MAX);
  if (pwmL>0 && pwmL<PWM_MIN) pwmL=PWM_MIN;
  if (pwmR>0 && pwmR<PWM_MIN) pwmR=PWM_MIN;

  if (cmdL >= 0){ digitalWrite(L_IN1,HIGH); digitalWrite(L_IN2,LOW); }
  else          { digitalWrite(L_IN1,LOW ); digitalWrite(L_IN2,HIGH);}
  analogWrite(L_PWM, pwmL);

  if (cmdR >= 0){ digitalWrite(R_IN1,HIGH); digitalWrite(R_IN2,LOW); }
  else          { digitalWrite(R_IN1,LOW ); digitalWrite(R_IN2,HIGH);}
  analogWrite(R_PWM, pwmR);
}
void stopMotor(){ analogWrite(L_PWM,0); analogWrite(R_PWM,0); }
void reverseMs(unsigned long ms){ unsigned long t0=millis(); while(millis()-t0<ms){ setMotorLR(-0.5,-0.5,true); delay(5);} stopMotor(); }

void readSerialCmd(){
  static String buf;
  while (Serial.available()){
    char c = Serial.read();
    if (c=='\n' || c=='\r'){
      buf.trim();
      if (buf.length()>=2){
        char side = toupper(buf[0]);
        int idx = buf.substring(1).toInt();
        if ((side=='R' || side=='L') && idx>=0){
          targetSide = (side=='R')?SIDE_R:SIDE_L;
          targetIdx  = idx;
          currentIdx = 0;
          state = FOLLOW_MAIN;
          Serial.print(F("[CMD] side=")); Serial.print((targetSide==SIDE_R)?"R":"L");
          Serial.print(F(", idx=")); Serial.println(targetIdx);
          resetObjFSM();
        }
      }
      buf="";
    }else{
      buf += c;
    }
  }
}

// 오른쪽(정밀): 4단계 near 패턴
int holdLead=0, holdMid=0, holdTrail=0;
bool rLead=false, rMid=false, rTrail=false;

void resetObjFSM(){
  holdLead=holdMid=holdTrail=0;
  rLead=rMid=rTrail=false;
  // 왼쪽 간이용
  lNearHold=0; lFarHold=0; lOpen=false;
}

// 왼쪽 간이 카운트(Lf만 사용): near 유지 → far 유지 => +1
int  lNearHold=0, lFarHold=0;
bool lOpen=false;

void setup(){
  Serial.begin(115200);

#if HAS_LF
  pinMode(TRIG_LF, OUTPUT); pinMode(ECHO_LF, INPUT);
#endif
  pinMode(TRIG_RF, OUTPUT); pinMode(ECHO_RF, INPUT);
  pinMode(TRIG_RR, OUTPUT); pinMode(ECHO_RR, INPUT);
  pinMode(TRIG_F , OUTPUT); pinMode(ECHO_F , INPUT);

  pinMode(L_PWM, OUTPUT); pinMode(L_IN1, OUTPUT); pinMode(L_IN2, OUTPUT);
  pinMode(R_PWM, OUTPUT); pinMode(R_IN1, OUTPUT); pinMode(R_IN2, OUTPUT);

  stopMotor();
  Serial.println(F("RoboSort Station Mode ready (send R3 / L2 etc.)"));
}

void loop(){
  readSerialCmd();

  unsigned long now = millis();
  if (now - lastLoop < LOOP_DT_MS) return;
  lastLoop = now;

#if HAS_LF
  float Lf = readUS(TRIG_LF, ECHO_LF);  emaLf = ema(emaLf, Lf);
#else
  float Lf = 0.30; // placeholder
  emaLf = ema(emaLf, Lf);
#endif
  float Rf = readUS(TRIG_RF, ECHO_RF);  emaRf = ema(emaRf, Rf);
  float Rr = readUS(TRIG_RR, ECHO_RR);  emaRr = ema(emaRr, Rr);
  float Fd = readUS(TRIG_F , ECHO_F );  emaF  = ema(emaF , Fd);

  if (emaF < 0.15){ stopMotor(); return; }

  switch(state){
    case IDLE:
      stopMotor();
      break;

    case FOLLOW_MAIN: {
      // 오른쪽 벽/물체 기준 추종
      float e_right = (d_set_right - emaRf);     // 오른쪽 목표거리 유지
      float s_yaw   = (emaRr - emaRf);           // 평행 보정
      float u = Kp*e_right + Ky*s_yaw;
      float v = V_FWD;
      setMotorLR(v - u, v + u, false);

      // ---- 카운트 ----
      if (targetSide == SIDE_R){
        bool rfNear = (emaRf < d_near);
        bool rrNear = (emaRr < d_near);

        if (rfNear && !rrNear){ if(++holdLead>=N_frames){ rLead=true; } }
        else if(!rfNear && !rrNear){ // 초기 상태로 리셋
          holdLead=0; rLead=false;
        }

        if (rfNear && rrNear && rLead){ if(++holdMid>=N_frames){ rMid=true; } }

        if (!rfNear && rrNear && rMid){ if(++holdTrail>=N_frames){ rTrail=true; } }

        // CLEAR: Rf far, Rr far -> 1개 스테이션 통과
        if (!rfNear && !rrNear && rTrail){
          currentIdx++;
          resetObjFSM();
          Serial.print(F("[STN] R idx -> ")); Serial.println(currentIdx);
          if (currentIdx == targetIdx){
            state = ALIGN_AT_STATION;
            stopMotor();
            Serial.println(F("[STATE] ALIGN_AT_STATION (R)"));
          }
        }

      } else { // LEFT
#if HAS_LF
        bool lfNear = (emaLf < d_near);
        if (lfNear){ lNearHold++; lFarHold=0; }
        else       { lFarHold++;  lNearHold=0; }

        if (!lOpen && lNearHold>=N_frames){ lOpen=true; }
        if (lOpen && lFarHold>=N_frames){
          currentIdx++; lOpen=false; lNearHold=lFarHold=0;
          Serial.print(F("[STN] L idx -> ")); Serial.println(currentIdx);
          if (currentIdx == targetIdx){
            state = ALIGN_AT_STATION;
            stopMotor();
            Serial.println(F("[STATE] ALIGN_AT_STATION (L)"));
          }
        }
#else
        // Lf가 없다면 그냥 오른쪽 기준으로 계속 주행(명령 무시)
#endif
      }
      break;
    }

    case ALIGN_AT_STATION: {
      // 목표 스테이션의 길이 중심 옆에서 정지
      // 오른쪽: Rf≈Rr가 되도록 전/후진으로 중앙 맞춤 + 평행 보정
      if (targetSide == SIDE_R){
        bool nearF = (emaRf < d_near);
        bool nearR = (emaRr < d_near);

        // 먼저 해당 스테이션 구간 안으로 들어오기
        if (!(nearF && nearR)){
          // 아직 완전히 스테이션 옆이 아니면 천천히 전진
          setMotorLR(+V_ALIGN, +V_ALIGN, true);
          break;
        }

        // 중앙 정렬: Rf와 Rr 차로 앞/뒤 미세 이동
        float diff = emaRr - emaRf; // >0면 뒤가 더 멂(=앞쪽이 더 가까움)
        float yaw  = (emaRr - emaRf);
        float u_yaw = Ky * yaw;

        const float eps = 0.015; // 약 1.5cm 이내면 OK
        if (fabs(diff) <= eps){
          stopMotor();
          state = DROP;
          Serial.println(F("[STATE] DROP"));
        } else if (diff > eps){
          // 뒤가 더 멀다 → 조금 앞으로 (앞이 더 가까움 → 균형 만들려면 전진)
          setMotorLR(+V_ALIGN - u_yaw, +V_ALIGN + u_yaw, true);
        } else {
          // 앞이 더 멀다 → 조금 뒤로
          setMotorLR(-V_ALIGN - u_yaw, -V_ALIGN + u_yaw, true);
        }
      } else { // LEFT
#if HAS_LF
        // Lf만으로는 길이 중심 정밀 정렬이 어렵다 → 근접 유지 시간 기반 간이정렬
        // Lf가 near 상태를 M 프레임 유지하면 중심 근처로 판단
        static int holdCenter=0;
        bool lfNear = (emaLf < d_near);
        if (lfNear){
          holdCenter++;
          // 평행 보정은 오른쪽 센서로 간접(없으면 0)
          float yaw  = (emaRr - emaRf);
          float u_yaw = Ky * yaw;
          setMotorLR(+0.35 - u_yaw, +0.35 + u_yaw, true);
          if (holdCenter>= (N_frames+2)){
            stopMotor();
            state = DROP;
            holdCenter=0;
            Serial.println(F("[STATE] DROP (L,approx)"));
          }
        }else{
          // 아직 멀면 천천히 접근
          setMotorLR(+V_ALIGN, +V_ALIGN, true);
        }
#else
        // Lf 없으면 왼쪽 정렬 불가
        state = IDLE;
#endif
      }
      break;
    }

    case DROP:
      // TODO: 서보 경사판 제어 추가
      stopMotor();
      delay(500);
      state = EXIT_FROM_STATION;
      Serial.println(F("[STATE] EXIT_FROM_STATION"));
      break;

    case EXIT_FROM_STATION:
      // 살짝 후진해서 스테이션에서 이격
      reverseMs(T_back_ms);
      state = IDLE;     // 또는 FOLLOW_MAIN으로 복귀하여 다음 명령 대기
      currentIdx = 0;   // 다음 명령 준비
      Serial.println(F("[STATE] IDLE"));
      break;

    case DONE:
      stopMotor();
      break;
  }

  // 디버깅 필요시
  // Serial.print("Rf=");Serial.print(emaRf,2);
  // Serial.print(" Rr=");Serial.print(emaRr,2);
  // Serial.print(" Lf=");Serial.print(emaLf,2);
  // Serial.print(" F=");Serial.print(emaF,2);
  // Serial.print(" idx=");Serial.print(currentIdx);
  // Serial.print(" tgt=");Serial.print(targetIdx);
  // Serial.print(" side=");Serial.println((targetSide==SIDE_R)?'R':'L');
}
