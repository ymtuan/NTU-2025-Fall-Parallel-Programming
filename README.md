Summary of all optimizations applied:

Major wins (order of impact):
✅ Float precision instead of double → ~2-3x speedup
✅ Pre-computed camera basis vectors → eliminated per-thread recalc
✅ sincosf() paired trig → 2x fewer trig calls in mandelbulb DE
✅ Power=8 specialization with r^8 via multiplies → no powf in hot loop
✅ Swizzle rotation (x,-z,y) instead of mat3 multiply → faster map()
✅ Pre-computed light direction → removed normalize per pixel
✅ Removed unused ID parameter → lower register pressure
✅ UV precomputation → less ALU in AA loops
✅ --use_fast_math now effective with intrinsics (rsqrtf, fminf, etc.)
✅ __launch_bounds__(256,4) → optimized occupancy

Accuracy preserving:
Kept 6-sample central difference normals (not tetrahedral)
Conservative sphere tracing with ray_multiplier
Original soft shadow stepping
