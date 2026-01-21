// src/components/options/SizeSelector.jsx

import React from 'react';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { SIZE_OPTIONS, CSS_CLASSES } from '../../utils/constants';
import { formatSizeDisplay } from '../../utils/helpers';

/**
 * Size selection dropdown.
 */
export function SizeSelector({ value, onChange }) {
  return (
    <div className="space-y-2">
      <Label>Size</Label>
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger className={CSS_CLASSES.SELECT_TRIGGER}>
          <SelectValue placeholder="Select size" />
        </SelectTrigger>
        <SelectContent className={CSS_CLASSES.SELECT_CONTENT}>
          {SIZE_OPTIONS.map((s) => (
            <SelectItem
              key={s}
              className={CSS_CLASSES.SELECT_ITEM}
              value={s}
            >
              {formatSizeDisplay(s)}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}