let
  pkgs = import <nixpkgs> {};
in pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (python-pkgs: [
      python-pkgs.matplotlib
      python-pkgs.numpy
      python-pkgs.torch
    ]))
  ];
}
