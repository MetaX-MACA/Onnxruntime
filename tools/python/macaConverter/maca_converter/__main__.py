import sys
import maca_converter
from maca_converter.model_convert  import main, parse_args, set_using_wheel

if __name__=="__main__":
   set_using_wheel()
   args = parse_args()
   main(args)