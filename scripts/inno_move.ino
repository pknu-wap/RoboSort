#include <Servo.h> //arduino-ide로 실행 

// 서보모터
Servo servo1; // 5번
Servo servo2; // 6번

// 모터드라이버 (L298N)

// 왼쪽 바퀴
int ENA = 3;
int IN1 = 7;
int IN2 = 8;

// 오른쪽 바퀴
int ENB = 11;
int IN3 = 12;
int IN4 = 13;

int v = 105;  // 속도
int m = 2500; // 시간 
int s = 87; //각도

void setup()
{
  Serial.begin(9600);

  servo1.attach(5);
  servo2.attach(6);

  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);

  pinMode(ENB, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  stopAll();

  servo1.write(s);
  servo2.write(s);

  Serial.println("RoboSort Ready!");
}

void loop()
{
  if (Serial.available() > 0)
  {
    String boxNumStr = Serial.readStringUntil('\n');
    boxNumStr.trim();

    Serial.print("[TEST] 입력됨 → ");
    Serial.println(boxNumStr);

    if (boxNumStr.startsWith("100"))
    {
      tiltLeft();
      Serial.println("Done");
    }
    else if (boxNumStr.startsWith("200"))
    {
      moveForward(m);
      tiltLeft();
      moveBackward(m);
      Serial.println("Done");
    }
    else if (boxNumStr.startsWith("300"))
    {
      moveForward(2*m);
      tiltLeft();
      moveBackward(2*m);
      Serial.println("Done");
    }
    else if (boxNumStr.startsWith("400"))
    {
      tiltRight();
      Serial.println("Done");
    }
    else if (boxNumStr.startsWith("500"))
    {
      moveForward(m);
      tiltRight();
      moveBackward(m);
      Serial.println("Done");
    }
    else if (boxNumStr.startsWith("600"))
    {
      moveForward(2*m);
      tiltRight();
      moveBackward(2*m);
      Serial.println("Done");
    }
    else
    {
      Serial.println("[ERROR] 알 수 없는 명령");
    }

    Serial.println("-- 입력 대기중 --");
  }
}

void stopAll()
{
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  analogWrite(ENA, 0);

  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
  analogWrite(ENB, 0);

  delay(100);
}

void leftForward()
{
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  analogWrite(ENA, v);
}

void leftBackward()
{
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  analogWrite(ENA, v);
}

void rightForward()
{
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
  analogWrite(ENB, v);
}

void rightBackward()
{
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  analogWrite(ENB, v);
}

void forward()
{
  leftForward();
  rightForward();
}

void backward()
{
  leftBackward();
  rightBackward();
}

void moveForward(int duration)
{
  forward();
  delay(duration);
  stopAll();
}

void moveBackward(int duration)
{
  backward();
  delay(duration);
  stopAll();
}


void tiltRight()
{
  servo1.write(s + 30);
  servo2.write(s - 30);
  delay(2000);

  servo1.write(s);
  servo2.write(s);
  delay(1000);

}


void tiltLeft()
{
  servo1.write(s - 20);
  servo2.write(s + 20);
  delay(2000);

  servo1.write(s);
  servo2.write(s);
  delay(1000);

}
