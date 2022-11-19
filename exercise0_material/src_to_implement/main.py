from pattern import Checker, Circle, Spectrum
from generator import ImageGenerator

gen = ImageGenerator('./exercise_data/', './Labels.json', 25, [32, 32, 3], rotation=False, mirroring=False,
                     shuffle=False)
gen.show()


# circle = Circle(1000, 150, (250, 300))
# circle.draw()
# circle.show()
#
#
# checker = Checker(1000, 100)
# checker.draw()
# checker.show()
#
#
# spectrum = Spectrum(1000)
# spectrum.draw()
# spectrum.show()