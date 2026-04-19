// src/lib/slashCommands/chat.js — /chat slash command
//
// Registers once at module import. Sends a single chat turn to the active mode's
// LLM via useChatJob, streams the reply into a CHAT bubble, then switches the
// composer to chat routing mode.

import { register } from '../slashCommands.js';
import { MESSAGE_KINDS, MESSAGE_ROLES } from '../../utils/constants.js';
import { nowId } from '../../utils/helpers.js';

register('chat', {
  description: "Send a message to the active mode's LLM",
  enabled: (ctx) => Boolean(ctx.chatEnabled),
  disabledReason: (ctx) => `Chat not enabled for mode ${ctx.activeMode ?? 'unknown'}`,
  handler: ({ args, ctx }) => {
    if (!args) return false;

    if (!ctx.chatEnabled) {
      ctx.addMessage(ctx.createErrorMessage(`Chat not enabled for mode ${ctx.activeMode ?? 'unknown'}`));
      return false;
    }
    if (!ctx.wsConnected) {
      ctx.addMessage(ctx.createErrorMessage('Not connected to server'));
      return false;
    }

    const userId = nowId();
    const assistantId = nowId();

    ctx.addMessage([
      {
        id: userId,
        role: MESSAGE_ROLES.USER,
        kind: MESSAGE_KINDS.CHAT,
        text: args,
        ts: Date.now(),
      },
      {
        id: assistantId,
        role: MESSAGE_ROLES.ASSISTANT,
        kind: MESSAGE_KINDS.CHAT,
        text: '',
        streaming: true,
        jobId: null,
        ts: Date.now(),
      },
    ]);

    let rafBuffer = '';
    let rafPending = false;
    let terminated = false;

    const flushDelta = () => {
      const chunk = rafBuffer;
      rafBuffer = '';
      rafPending = false;
      if (chunk && !terminated) {
        ctx.updateMessage(assistantId, (prev) => ({ ...prev, text: prev.text + chunk }));
      }
    };

    const handle = ctx.chatJob.start({
      prompt: args,
      onAck: ({ jobId }) => {
        ctx.updateMessage(assistantId, { jobId });
      },
      onDelta: (text) => {
        if (terminated) return;
        rafBuffer += text;
        if (!rafPending) {
          rafPending = true;
          requestAnimationFrame(flushDelta);
        }
      },
      onComplete: ({ text }) => {
        terminated = true;
        ctx.updateMessage(assistantId, { text, streaming: false, jobId: null });
      },
      onError: (errMsg) => {
        terminated = true;
        ctx.updateMessage(assistantId, (prev) => ({
          ...prev,
          kind: MESSAGE_KINDS.ERROR,
          prevText: prev.text,
          text: errMsg,
          streaming: false,
          jobId: null,
        }));
      },
    });

    ctx.updateMessage(assistantId, { cancelHandle: handle });
    ctx.setInputMode('chat');
    return true;
  },
});
