from pygame.time import Clock
from articulate.utils.pygame import StreamingDataViewer
from articulate.utils.wearable import WearableSensorSet


def test_wearable_pressure():
    clock = Clock()
    sviewer = StreamingDataViewer(1, y_range=(0, 10), window_length=200); sviewer.connect()
    sensor_set = WearableSensorSet()
    while True:
        data = sensor_set.get()
        if 0 in data.keys():
            sviewer.plot([data[0].pressure])
            clock.tick(60)
            print('\r', clock.get_fps(), end='')
            
if __name__ == '__main__':
    test_wearable_pressure()