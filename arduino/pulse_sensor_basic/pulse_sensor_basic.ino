int PulseSensorPurplePin1 = 4;
// int PulseSensorPurplePin2 = 14;
int Signal1;
// int Signal2;
//int Threshold = 580;
void setup(){
  Serial.begin(115200);
};
void loop(){
  Signal1 = analogRead(PulseSensorPurplePin1);
  // Signal2 = analogRead(PulseSensorPurplePin2);  
  Serial.print(Signal1); 
  Serial.print(", ");
  // Serial.print(Signal2); 
  // Serial.print(", ");
  Serial.print(int(millis()/1000));
  Serial.println();
delay(10);
};