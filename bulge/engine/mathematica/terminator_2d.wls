#!/usr/bin/env wolframscript
(* ::Package:: *)

planet = pxx x^2 + pxy x y + pxz x z + px0 x + pyy y^2 + pyz y z + + py0 y + pzz z^2 + pz0 z + p00;
coeffs = FullSimplify[CoefficientRules[planet /. Solve[Grad[planet, {x,y,z}] . {xc,yc,zc}  == 0, z][[1]], {x,y}]];

files = {
    "terminator_coeff_txx.txt",
    "terminator_coeff_txy.txt",
    "terminator_coeff_tx0.txt",
    "terminator_coeff_tyy.txt",
    "terminator_coeff_ty0.txt",
    "terminator_coeff_t00.txt"}

Do[
Export[files[[v]],
 FortranForm[Simplify[coeffs[[v]][[2]]]]], {v,1,6}]
