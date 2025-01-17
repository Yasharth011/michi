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
        packages.ompl = with pkgs;
        stdenv.mkDerivation {
          name = "ompl";
          src = fetchFromGitHub {
            owner = "ompl";
            repo = "ompl";
            rev = "e2994e580fcba0eb117e38bd1dc7439c4beb35ef";
            sha256 = "sha256-6jsB7uZThGvnMACmGE2k1v0aLnudJsWNduAJ2VAH2Oo=";
          };
          nativeBuildInputs = [cmake pkg-config];
          buildInputs = [boost eigen];
          configurePhase = ''
            mkdir build && cd build
            cmake .. -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_INSTALL_PREFIX=$out \
            -DOMPL_BUILD_TESTS=OFF -DOMPL_VERSIONED_INSTALL=OFF -DOMPL_REGISTRATION=OFF
          '';
          installPhase = ''
            make install
          '';
          meta = {
            description = "ompl";
          };
        };
        packages.opencv = pkgs.opencv.override {
          enableGtk2 = true;
        };
         packages.ixwebsocket = with pkgs;
         stdenv.mkDerivation {
           name = "ixwebsocket";
           src = fetchFromGitHub {
             owner = "machinezone";
             repo = "IXWebSocket";
             rev = "c5a02f1066fb0fde48f80f51178429a27f689a39";
             sha256 = "sha256-j/Fa45es2Oj09ikMZ8rMsSEOXLgO5+H7aqNurOua9LY=";
            };
            patches = [(fetchpatch {
              # Need to patch CMakeLists for using SpdLog from propagated inputs
              url = "https://github.com/kknives/IXWebSocket/commit/73f5d8d4cec5a336f642a03d2d067cd8acee17dc.patch";
              hash = "sha256-RulJ2k6FF0X/d4ZVWBIf2gXmLMhVRQQeRjscDdhzNdk=";
            })];
           nativeBuildInputs = [cmake pkg-config spdlog];
           propagatedBuildInputs = [
             openssl.dev
             zlib.dev
             curl.dev
           ];
           configurePhase = ''
             mkdir build && cd build
             cmake .. -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_INSTALL_PREFIX=$out \
             -DBUILD_SHARED_LIBS=ON -DUSE_ZLIB=1 -DUSE_TLS=1 -DUSE_WS=1
           '';
           installPhase = ''
             make install
           '';
           meta = {
             description = "ixwebsocket";
           };
         };
        packages.morb_slam = with pkgs;
        stdenv.mkDerivation {
          name = "morb_slam";
          src = fetchFromGitHub {
            owner = "Soldann";
            repo = "MORB_SLAM";
            rev = "9bfbb0b2d34951604c2a0d929e841df2fb9a66a3";
            sha256 = "sha256-Vyve3Scc9wmeLV45ebcaWa6u2j32eM8pDTBuoK/M20I=";
          };
          patches = [(fetchpatch {
            url = "https://gist.githubusercontent.com/kknives/2cef9d28800f46967dd3a8df17f8dfa1/raw/3d61d7798fe672a22ff45ba1925f460c801c27cd/fix_dirs.patch";
            hash = "sha256-aHI9DCh8+Zrlb5EOiM0qJmRilPHSr5OyQoB8rarpC9o=";
          })];
          nativeBuildInputs = [cmake pkg-config copyPkgconfigItems];
          pkgconfigItems = [
            (makePkgconfigItem rec {
              name = "MORB_SLAM";
              version = "3.0.0";
              cflags = [ "-I${variables.cameradir} -I${variables.dbow2dir} -I${variables.sophusdir} -I${variables.g2odir} -I${variables.includedir}"];
              libs = [ "-L${variables.lddir} -lMORB_SLAM -L${variables.dbow2lddir} -lDBoW2 -L${variables.g2olddir} -lg2o"];
              variables = rec {
                prefix = "${placeholder "out"}";
                includedir = "${prefix}/include/source/include";
                cameradir = "${prefix}/include/source/include/CameraModels";
                dbow2dir = "${prefix}/include/source/Thirdparty/DBoW2";
                sophusdir = "${prefix}/include/source/Thirdparty/Sophus";
                g2odir = "${prefix}/include/source/Thirdparty/g2o";
                lddir = "${prefix}/lib";
                dbow2lddir = "${prefix}/include/source/build/Thirdparty/DBoW2/lib";
                g2olddir = "${prefix}/include/source/build/Thirdparty/g2o/lib";
              };
            })
          ];
          passthru.tests.pkg-config = testers.hasPkgConfigModule {
            package = finalAttrs.finalPackage;
            moduleName = "MORB_SLAM";
          };
          buildInputs = [eigen pangolin packages.ixwebsocket glew gdal
           packages.opencv boost];
          # configurePhase = ''
          # ./build.sh
          # '';
          configurePhase = ''
          mkdir build && cd build
          mkdir -p Thirdparty/DBoW2 Thirdparty/g2o Thirdparty/Sophus

          cd Thirdparty/DBoW2
          cmake ../../../Thirdparty/DBoW2 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_INSTALL_PREFIX=$out
          make -j
          mkdir -p $out/lib/Thirdparty/DBoW2/lib
          # cp /build/source/Thirdparty/DBoW2/lib/libDBoW2.so $out/lib/Thirdparty/DBoW2/lib

          cd ../g2o
          cmake ../../../Thirdparty/g2o -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_INSTALL_PREFIX=$out
          make -j
          mkdir -p $out/lib/Thirdparty/g2o/lib
          # cp /build/source/Thirdparty/g2o/lib/libg2o.so $out/lib/Thirdparty/g2o/lib

          cd ../Sophus
          cmake ../../../Thirdparty/Sophus -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_INSTALL_PREFIX=$out -DBUILD_TESTS=OFF
          make -j

          cd ../..
          cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_INSTALL_PREFIX=$out
          '';
          buildPhase = ''
          make -j6
          '';
          installPhase = ''
          runHook preInstall
          make install
          rm -f $out/include/source/build/libMORB_SLAM.so
          mkdir -p $out/include/CameraModels
          mv $out/include/include/CameraModels/* $out/include/CameraModels
          rm -rf $out/include/include/CameraModels
          mv $out/include/include/* $out/include
          runHook postInstall
          # prev_rpath=$(patchelf --print-rpath libMORB_SLAM.so | sed 's#/build/source#'$out/lib#g) 
          # echo "Modified rpath=$prev_rpath"
          # patchelf --set-rpath $prev_rpath libMORB_SLAM.so
          '';
          meta = {
            description = "morb_slam";
          };
        };
        packages.michi = with pkgs;
          stdenv.mkDerivation {
            name = "michi";
            src = self;
            nativeBuildInputs = [ cmake gcc13 pkg-config ccache ];
            buildInputs = [
              packages.librealsense.dev
              eigen
              pcl
              boost.dev
              packages.opencv
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
              packages.cobs-c
              octomap
              packages.fusion
              packages.ompl
              gmp
              mpfr
              cgal
              glew
              pangolin
              packages.morb_slam
              packages.ixwebsocket
            ];
            configurePhase = ''
              cmake -S . -B build -DBUILD_EKF_GZ=OFF -DBUILD_TESTS=OFF \
                -DCMAKE_BUILD_TYPE=Release
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
