(* ::Package:: *)

(* Mathematica/Wolfram Language workflow for the two-pathway enzyme example. *)
(* Species order: x1=s, x2=e, x3=c1, x4=c2, x5=c, x6=p. *)
(* Reversible catalytic step: X5 <-> X2 + X6 with rates k5, km5. *)

ClearAll["Global`*"];

xvars = {x1, x2, x3, x4, x5, x6};
kvars = {k1, k2, k3, k4, k5, km1, km2, km3, km4, km5};
tvars = {t1, t2, t3, t4, t5, t6, t7, t8, t9};

T = {{1, t1, 0, 0, t2, t3},
     {0, t4, 1, 0, t5, t6},
     {0, t7, 0, 1, t8, t9}};

B = {{t1, t2, t3},
     {-1, 0, 0},
     {t4, t5, t6},
     {t7, t8, t9},
     {0, -1, 0},
     {0, 0, -1}};

kernelCheck = Simplify[T . B];
If[kernelCheck =!= ConstantArray[0, {3, 3}],
   Print["ERROR: B is not a right kernel basis for T: ", kernelCheck]];

(* Reversible mass-action vector field, with the + km5*x2*x6 term in x5d. *)
x1d = -k1*x1*x2 + km1*x3 - k3*x1*x2 + km3*x4;
x2d = -k1*x1*x2 + km1*x3 - k3*x1*x2 + km3*x4 + k5*x5 - km5*x2*x6;
x3d =  k1*x1*x2 - km1*x3 - k2*x3 + km2*x5;
x4d =  k3*x1*x2 - km3*x4 - k4*x4 + km4*x5;
x5d =  k2*x3 + k4*x4 - km2*x5 - km4*x5 - k5*x5 + km5*x2*x6;
x6d =  k5*x5 - km5*x2*x6;

F = {x1d, x2d, x3d, x4d, x5d, x6d};

(* Conservation checks for the reversible enzyme model. Both should be exactly zero. *)
totalEnzymeCheck = Factor[x2d + x3d + x4d + x5d];
totalSubstrateProductCheck = Factor[x1d + x3d + x4d + x5d + x6d];
If[totalEnzymeCheck =!= 0, Print["ERROR: total enzyme is not conserved: ", totalEnzymeCheck]];
If[totalSubstrateProductCheck =!= 0, Print["ERROR: total substrate/product balance is not conserved: ", totalSubstrateProductCheck]];

jac = D[F, {xvars}];
U = FullSimplify[T . jac . B];

coefficientsInStateVariables[expr_] := Values[CoefficientRules[Expand[expr], xvars]];
conditions = Factor[DeleteCases[Flatten[coefficientsInStateVariables /@ Flatten[U]], 0]];

arrays = CoefficientArrays[conditions, kvars];
constantPart = Normal[arrays[[1]]];
M = Normal[arrays[[2]]];

If[Factor[constantPart] =!= ConstantArray[0, Length[conditions]],
   Print["ERROR: conditions are not homogeneous-linear in rate constants. Constant part: ", constantPart]];

(* The km5-sensitive conditions arising from the reversible final step. *)
km5SensitiveConditions = Select[conditions, ! FreeQ[#, km5] &] // Factor;

Print["T.B = "]; Print[MatrixForm[kernelCheck]];
Print["Total enzyme conservation check = ", totalEnzymeCheck];
Print["Total substrate/product conservation check = ", totalSubstrateProductCheck];
Print["Number of coefficient conditions = ", Length[conditions]];
Print["km5-sensitive conditions:"]; Print[Column[km5SensitiveConditions]];
Print["Coefficient matrix M has dimensions ", Dimensions[M]];



