# 🦾RoboSort: 스마트 택배 분류 로봇

> **RoboSort**는 **운송장 번호를 OCR 인식**하고, **해당 구역으로 물품을 분류하는 로봇 시스템**입니다.  
> Python 기반 OCR과 Arduino 제어를 결합해 **실시간 인식 및 제어 자동화**를 구현했습니다.

---

## 🧩 프로젝트 개요  
- **목적**: 물류 현장의 반복적인 수작업 분류 과정을 자동화  
- **핵심 아이디어**:  
  - 웹캠으로 운송장 촬영  
  - OCR을 통해 번호 인식  
  - 인식된 번호의 백의 자리 기준으로 구역 자동 분류  
  - 아두이노 제어로 서보모터 회전 및 물품 이동

---

## ⚙️ 기술 스택

| 분야 | 사용 기술 |
|------|------------|
| 하드웨어 | Arduino Uno, Servo Motor, 초음파 센서, 웹캠 |
| 소프트웨어 | Python 3.13, OpenCV, RapidOCR, PySerial |
| 모델 | RapidOCR (ONNX 기반 문자 인식 모델) |
| 통신 방식 | Serial (USB 기반 데이터 송수신) |

---

## 👥 팀원 소개

|  |  |
|--|--|
| <img src="https://github.com/hheyyeon21.png" width="100" height="100" /> | **이름:** 문희연<br>**역할:** 팀장<br>**담당 내용:** 프로젝트 총괄<br>
| <img src="https://github.com/wave1009.png" width="100" height="100" /> | **이름:** 김물결<br>**역할:** 소프트웨어<br>**담당 내용:** Python OCR 로직 및 RapidOCR 적용<br>
| <img src="https://github.com/leeseungmin313.png" width="100" height="100" /> | **이름:** 이승민<br>**역할:** 하드웨어<br>**담당 내용:** 아두이노 서보모터 및 센서 제어<br>
| <img src="https://github.com/jisuuuu21.png" width="100" height="100" /> | **이름:** 김지수<br>**역할:** 설계<br>**담당 내용:** 전체 설계 및 회로 다이어그램 디자인<br>
