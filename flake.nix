{
  description = "Make your rover autonomous";
  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url =
      "github:NixOS/nixpkgs/e1ad98971a8b3994be8b90c27a7f0790cb5da51c";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let pkgs = nixpkgs.legacyPackages.${system};
      in rec {
        packages.librealsense = pkgs.librealsense.overrideAttrs
          (finalAttrs: previousAttrs: {
            postInstall = ''
              substituteInPlace $out/lib/cmake/realsense2/realsense2Targets.cmake \
                --replace "\''${_IMPORT_PREFIX}/include" "$dev/include"
            '';
          });
          packages.mavlink_c = with pkgs;
            stdenv.mkDerivation {
              name = "mavlink_c_library_v2";
              src = fetchFromGitHub {
                owner = "mavlink";
                repo = "c_library_v2";
                rev = "494fd857a34267d01c2d3c2d601ecfa651f73489";
                sha256 = "sha256-FiuD9G+1sSYfBFpTCw6c5mnpFbDkZJwYFYtL3o1ujAo=";
              };
              nativeBuildInputs = [ copyPkgconfigItems ];
              pkgconfigItems = [
                (makePkgconfigItem rec {
                  name = "mavlink_c";
                  version = "2";
                  cflags = [ "-I${variables.includedir}/mavlink"];
                  variables = rec {
                    prefix = "${placeholder "out"}";
                    includedir = "${prefix}/include";
                  };
                })
              ];
              dontBuild = true;
              installPhase = ''
              runHook preInstall
              mkdir -p $out/include/mavlink
              cp -R standard common minimal $out/include/mavlink
              cp *.h $out/include/mavlink
              runHook postInstall
              '';
            };
        packages.gz_cmake = with pkgs;
          stdenv.mkDerivation {
            name = "gz_cmake";
            src = fetchFromGitHub {
              owner = "gazebosim";
              repo = "gz-cmake";
              rev = "459ea347de71d5a9cd0ef2cf13fe14e6421ecb6e";
              sha256 = "sha256-cmEdtGQ2h3eelRbyr9MLCrkI/phoqaCSA7wv2fW+ylo=";
            };
            nativeBuildInputs = [cmake];
            prePatch = ''
              substituteInPlace config/gz-cmake.pc.in \
                --replace '$'{prefix}/@CMAKE_INSTALL_LIBDIR@ @CMAKE_INSTALL_FULL_LIBDIR@ \
                --replace '$'{prefix}/@CMAKE_INSTALL_INCLUDEDIR@ @CMAKE_INSTALL_FULL_INCLUDEDIR@
                '';
            configurePhase = ''
                mkdir build && cd build
                cmake .. -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_INSTALL_PREFIX=$out
              '';

              installPhase = ''
                make install
              '';

              meta = {
                description = "gz-cmake3";
              };
          };
          packages.gz_utils = with pkgs;
          stdenv.mkDerivation {
            name = "gz_utils";
            src = fetchFromGitHub {
              owner = "gazebosim";
              repo = "gz-utils";
              rev = "8f17f54c48ac21382f9f13776e66906046cc3d94";
              sha256 = "sha256-osY+q+H7F05gcLrpyMGeLsddh2nevG4lZsFeyeZWdaY=";
            };
            nativeBuildInputs = [cmake];
            buildInputs = [packages.gz_cmake];
            configurePhase = ''
                mkdir build && cd build
                cmake .. -DCMAKE_INSTALL_PREFIX=$out -DBUILD_TESTING=OFF
              '';

              installPhase = ''
              runHook preInstall
                cmake --install .
              runHook postInstall
              '';
              meta = {
                description = "gz-utils2";
              };
          };
        packages.gz_math = with pkgs;
          stdenv.mkDerivation {
            name = "gz_math";
            src = fetchFromGitHub {
              owner = "gazebosim";
              repo = "gz-math";
              rev = "39e48c1388e30a1eac101bc89d34937a394d7d95";
              sha256 = "sha256-Gj69j3PH4AlSMvzd3OjPF5wmQ0PfDKDtSH3CIgdFxBg=";
            };
            nativeBuildInputs = [cmake pkg-config];
            buildInputs = [eigen packages.gz_cmake packages.gz_utils];
            configurePhase = ''
                mkdir build && cd build
                cmake .. -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_INSTALL_PREFIX=$out \
                -DBUILD_TESTING=OFF
              '';

              installPhase = ''
                make install
              '';
              meta = {
                description = "gz-math7";
              };
          };
        packages.gz_msgs = with pkgs;
          stdenv.mkDerivation {
            name = "gz_msgs";
            src = fetchFromGitHub {
              owner = "gazebosim";
              repo = "gz-msgs";
              rev = "0472ba0bb5fe39d8a14499155c68746109d9acf7";
              sha256 = "sha256-wRbvGJAjwUD4YMlvgP70DytKGrPEhhxtIUcaLPkZ68I=";
            };
            nativeBuildInputs = [cmake pkg-config];
            buildInputs = [protobuf packages.gz_math packages.gz_cmake packages.gz_utils python3 tinyxml-2];
            configurePhase = ''
                mkdir build && cd build
                cmake .. -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_INSTALL_PREFIX=$out \
                -DBUILD_TESTING=OFF
              '';

              installPhase = ''
                make install
              '';
              meta = {
                description = "gz-msgs10";
              };
          };
        packages.gz_transport = with pkgs;
          stdenv.mkDerivation {
            name = "gz_transport";
            src = fetchFromGitHub {
              owner = "gazebosim";
              repo = "gz-transport";
              rev = "cfb80d25904920fe86ae8b1ca50b8fe46788d00f";
              sha256 = "sha256-+jEkBeXujnChYemWt+XwCE8CqLpMpnc7nP4vl8C3kOQ=";
            };
            nativeBuildInputs = [cmake pkg-config];
            propagatedBuildInputs = [
              protobuf
              libuuid.dev
              zeromq
              tinyxml-2
              python3
              sqlite
              cppzmq
              libsodium
              packages.gz_msgs
              packages.gz_math
              packages.gz_cmake
              packages.gz_utils
            ];
              meta = {
                description = "gz-transport13";
              };
          };
        packages.behaviortree_cpp = with pkgs;
          stdenv.mkDerivation {
            name = "BehaviourTree.CPP";
            src = fetchFromGitHub {
              owner = "BehaviorTree";
              repo = "BehaviorTree.CPP";
              rev = "ec2e6965732f4840697685ca4672ed754fde395a";
              sha256 = "sha256-U7hzCXlsfsUuyf8mm2I532RfAzrd6OjzWVbHXtMjwvA=";
            };
            nativeBuildInputs = [cmake pkg-config];
            propagatedBuildInputs = [
            zeromq
            sqlite
            ];
            configurePhase = ''
              mkdir build && cd build
              cmake .. -DBTCPP_UNIT_TESTS=OFF -DBTCPP_EXAMPLES=OFF \
                -DBTCPP_SQLITE_LOGGING=ON -DBTCPP_GROOT_INTERFACE=ON \
                -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_INSTALL_PREFIX=$out
             '';
            meta = {
              description = "BehaviorTree.CPP";
            };
          };
        packages.cobs-c = with pkgs;
          stdenv.mkDerivation {
            name = "cobs-c";
            src = fetchFromGitHub {
              owner = "cmcqueen";
              repo = "cobs-c";
              rev = "6cc55cddb06568bc026ed85f8e5f433496a3622c";
              sha256 = "sha256-aIWT5w3KUHEzxiWuHlfNWuxvjuCGX2nCBFYHNmYc2Is=";
            };
            nativeBuildInputs = [pkg-config validatePkgConfig autoreconfHook];
            passthru.tests.pkg-config = testers.hasPkgConfigModule {
              package = finalAttrs.finalPackage;
              moduleName = "cobs";
            };
          };
        packages.fusion = with pkgs;
          stdenv.mkDerivation {
            name = "fusion";
            src = fetchFromGitHub {
              owner = "xioTechnologies";
              repo = "Fusion";
              rev = "e7d2b41e6506fa9c85492b91becf003262f14977";
              sha256 = "sha256-c9YSCxUKYlpQDYj2Mb6H1F6+2ZP5U9rNrX6GiexXF5c=";
            };
            nativeBuildInputs = [pkg-config cmake copyPkgconfigItems ];
            pkgconfigItems = [
              (makePkgconfigItem rec {
                name = "Fusion";
                version = "1.2.5";
                cflags = [ "-I${variables.includedir}" ];
                libs = [ "-L${variables.lddir}" "-lFusion"];
                variables = rec {
                  prefix = "${placeholder "out"}";
                  includedir = "${prefix}/include";
                  lddir = "${prefix}/lib";
                };
              })
            ];
            passthru.tests.pkg-config = testers.hasPkgConfigModule {
              package = finalAttrs.finalPackage;
              moduleName = "Fusion";
            };
            configurePhase = ''
              mkdir build && cd build
              cmake .. -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_INSTALL_PREFIX=$out
            '';
            installPhase = ''
            runHook preInstall
            mkdir -p $out/lib
            mv Fusion/* $out/lib
            mkdir -p $out/include/Fusion
            cp -t $out/include/Fusion ../Fusion/*.h
            runHook postInstall
            '';
            meta = {
              description = "Fusion";
            };
          };
        packages.michi = with pkgs;
          stdenv.mkDerivation {
            name = "michi";
            src = self;
            nativeBuildInputs = [ cmake gcc13 pkg-config ];
            buildInputs = [
              packages.librealsense.dev
              eigen
              pcl
              boost.dev
              opencv
              glfw
              libGLU.dev
              spdlog.dev
              asio
              gtest.dev
              onnxruntime.dev
              packages.mavlink_c
              argparse
              packages.gz_transport
              packages.gz_msgs
              packages.behaviortree_cpp
              packages.cobs-c
              octomap
              packages.fusion
            ];
            configurePhase = ''
              cmake -S . -B build
            '';
            preBuild = ''
              cd build
            '';
            installPhase = ''
              mkdir -p "$out/bin"
              cp -r ../build $out/bin
            '';
          };
        defaultPackage = packages.michi;
      });
}
