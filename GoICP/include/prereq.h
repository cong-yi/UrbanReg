#if defined _WIN32 || defined __CYGWIN__
  #ifdef GOICP_EXPORT
    #ifdef __GNUC__
      #define GOICP_PUBLIC __attribute__ ((dllexport))
    #else
      #define GOICP_PUBLIC __declspec(dllexport) // Note: actually gcc seems to also supports this syntax.
    #endif
  #else
    #ifdef __GNUC__
      #define GOICP_PUBLIC __attribute__ ((dllimport))
    #else
      #define GOICP_PUBLIC __declspec(dllimport) // Note: actually gcc seems to also supports this syntax.
    #endif
  #endif
  #define GOICP_LOCAL
#else
  #if __GNUC__ >= 4
    #define GOICP_PUBLIC __attribute__ ((visibility ("default")))
    #define GOICP_LOCAL  __attribute__ ((visibility ("hidden")))
  #else
    #define GOICP_PUBLIC
    #define GOICP_LOCAL
  #endif
#endif