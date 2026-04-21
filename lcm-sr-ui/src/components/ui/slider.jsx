import * as React from "react";
import * as SliderPrimitive from "@radix-ui/react-slider";
import { cn } from "@/lib/utils";

const Slider = React.forwardRef(({ className, ...props }, ref) => (
  <SliderPrimitive.Root
    ref={ref}
    className={cn("relative flex w-full touch-none select-none items-center", className)}
    {...props}
  >
    <SliderPrimitive.Track className="relative h-2 w-full grow overflow-hidden rounded-full bg-gray-200 dark:bg-zinc-700">
      <SliderPrimitive.Range className="absolute h-full bg-indigo-600" />
    </SliderPrimitive.Track>
    <SliderPrimitive.Thumb className="block h-5 w-5 rounded-full border-2 border-indigo-600 bg-white shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400" />
  </SliderPrimitive.Root>
));
Slider.displayName = "Slider";

export { Slider };
