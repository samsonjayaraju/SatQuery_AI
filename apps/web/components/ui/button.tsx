import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap transition-colors disabled:pointer-events-none disabled:opacity-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#53e1c2]",
  {
    variants: {
      variant: {
        default: "bg-[#53e1c2] text-[#08201c] hover:bg-[#79ead3]",
        outline: "border border-[#2c514c] bg-[#0d2221] text-[#8fc5bd] hover:border-[#1d8c7b] hover:text-[#d9f6f1]",
        ghost: "bg-transparent text-[#78908c] hover:bg-[#102624]",
      },
      size: { default: "h-9 px-4 text-xs", sm: "h-8 px-3 text-[10px]", icon: "size-8" },
    },
    defaultVariants: { variant: "default", size: "default" },
  },
);

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement>, VariantProps<typeof buttonVariants> {}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(({ className, variant, size, ...props }, ref) => (
  <button className={cn(buttonVariants({ variant, size }), className)} ref={ref} {...props} />
));
Button.displayName = "Button";

export { buttonVariants };
