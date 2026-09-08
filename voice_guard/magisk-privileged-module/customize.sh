# Magisk module install script — runs inside the Magisk Manager install
# context, after $MODPATH/system/* has already been staged from this
# module's zip by the generic installer, before magic-mount takes effect.
#
# Only job here: fix ownership/permissions on the staged files to match
# what a real /system partition expects, since files added via a module zip
# don't automatically get sane system perms. Magisk auto-mounts anything
# under $MODPATH/system, so no explicit mount step is needed.

ui_print "- Setting permissions on VoiceGuard priv-app files"

set_perm_recursive "$MODPATH/system/priv-app/VoiceGuard" 0 0 0755 0644

set_perm "$MODPATH/system/etc/permissions/privapp-permissions-com.voiceguard.voice_guard.xml" 0 0 0644

if [ ! -f "$MODPATH/system/priv-app/VoiceGuard/VoiceGuard.apk" ]; then
  ui_print "! VoiceGuard.apk is missing from this module."
  ui_print "! Build it first with build_module.sh — see README.md in this"
  ui_print "! module's source directory. Aborting install."
  abort "VoiceGuard.apk not found in module payload"
fi

ui_print "- VoiceGuard privileged capture module staged. Reboot to activate."
