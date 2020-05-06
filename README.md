# Q-learning
Ten projekt jest jednym z moich starych projektów, w ramach którego starałem się nauczyć komputer grać w pewną grę z kolekcji atari. Także udostępniłem możliwość grania samemu
za pomocą klawiatury.
## Algorithm
Uczenie maszynowe wykorzystuję algorytm Q-lerning, czyli jeżeli piłka została odbita, dajemy nagrodę w przeciwnym wypadku - karę. Pozycje platformy i piłki otrzymuję na podstawie 2 obrazów (aktualnego i poprzedniego). Porównując te obrazy, możemy
zaobserwować w pewnym punkcie niezgodność, co oznacza, że w tym miejscu znajduje się obiekt.